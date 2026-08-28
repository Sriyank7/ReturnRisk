using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ReturnRisk.Core;

public class OrderData
{
    public bool Label { get; set; }
    public float Day { get; set; }
    [VectorType]
    public float[] Features { get; set; } = Array.Empty<float>();
}

public static class FeedbackLoopSimulation
{
    public static List<double> RunFeedbackLoop(
        MLContext ml,
        Experiments experiments,
        IDataView seedTrain,
        List<IDataView> rounds,
        IDataView test,
        MerchantProfile profile,
        int priceFeatureIndex,
        double medianRawPrice,
        double holdoutFraction,
        bool oracle,
        int seed)
    {
        var rng = new Random(seed);
        var aucHistory = new List<double>();
        double priceScaleFactor = profile.MedianOrderValue / medianRawPrice;

        // 1. Dynamically infer the exact feature vector size from seedTrain schema
        var featureVectorType = (VectorDataViewType)seedTrain.Schema["Features"].Type;
        int featureCount = featureVectorType.Size;

        var schemaDef = SchemaDefinition.Create(typeof(OrderData));
        schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);

        // 2. Materialize seed training data
        var accumulatedRows = ml.Data.CreateEnumerable<OrderData>(seedTrain, reuseRowObject: false).ToList();

        // 3. Baseline evaluation on seed data
        var currentTrainView = ml.Data.LoadFromEnumerable(accumulatedRows, schemaDef);
        var pipeline = experiments.BuildPipeline();
        var model = pipeline.Fit(currentTrainView);

        var testPreds = model.Transform(test);
        var testMetrics = ml.BinaryClassification.Evaluate(testPreds, labelColumnName: "Label");
        aucHistory.Add(testMetrics.AreaUnderRocCurve);

        // 4. Sequential simulation rounds
        for (int r = 0; r < rounds.Count; r++)
        {
            var roundView = rounds[r];

            var scoredRound = model.Transform(roundView);
            var probs = scoredRound.GetColumn<float>("Probability").ToArray();
            var roundRows = ml.Data.CreateEnumerable<OrderData>(roundView, reuseRowObject: false).ToArray();

            var observedRowsThisRound = new List<OrderData>();

            for (int i = 0; i < roundRows.Length; i++)
            {
                if (oracle)
                {
                    observedRowsThisRound.Add(roundRows[i]);
                }
                else
                {
                    double rawItemPrice = roundRows[i].Features[priceFeatureIndex];
                    double scaledItemPrice = rawItemPrice * priceScaleFactor;
                    double tau = CostModel.Tau(scaledItemPrice, profile);

                    bool wouldBlock = probs[i] > tau;

                    if (!wouldBlock)
                    {
                        observedRowsThisRound.Add(roundRows[i]);
                    }
                    else if (rng.NextDouble() < holdoutFraction)
                    {
                        observedRowsThisRound.Add(roundRows[i]);
                    }
                }
            }

            // Append observed rows and reload with explicit fixed-size SchemaDefinition
            accumulatedRows.AddRange(observedRowsThisRound);
            currentTrainView = ml.Data.LoadFromEnumerable(accumulatedRows, schemaDef);

            // Refit once per round and evaluate
            model = pipeline.Fit(currentTrainView);
            testPreds = model.Transform(test);
            testMetrics = ml.BinaryClassification.Evaluate(testPreds, labelColumnName: "Label");

            aucHistory.Add(testMetrics.AreaUnderRocCurve);
        }

        return aucHistory;
    }
}