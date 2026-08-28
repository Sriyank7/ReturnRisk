using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace ReturnRisk.Core;

public record ExperimentResult(string Name, double Auc, double PrAuc);
public record ReliabilityBin(int Bin, int Count, double MeanPredicted, double Observed, double Gap);

public class Experiments
{
    private readonly MLContext _ml;

    public Experiments(MLContext mlContext)
    {
        _ml = mlContext;
    }

    public IEstimator<ITransformer> BuildPipeline()
    {
        var options = new LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = "Features",
            NumberOfIterations = 300,
            LearningRate = 0.05,
            NumberOfLeaves = 31,
            Seed = 42
        };

        return _ml.BinaryClassification.Trainers.LightGbm(options);
    }

    public ExperimentResult RunTemporal(string name, IDataView train, IDataView validation)
    {
        var pipeline = BuildPipeline();
        var model = pipeline.Fit(train);
        var predictions = model.Transform(validation);
        var metrics = _ml.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

        return new ExperimentResult(name, metrics.AreaUnderRocCurve, metrics.AreaUnderPrecisionRecallCurve);
    }

    public ExperimentResult RunRandomCv(string name, IDataView allData)
    {
        var pipeline = BuildPipeline();
        var results = _ml.BinaryClassification.CrossValidate(allData, pipeline, numberOfFolds: 5, labelColumnName: "Label");

        double avgAuc = results.Average(r => r.Metrics.AreaUnderRocCurve);
        double avgPrAuc = results.Average(r => r.Metrics.AreaUnderPrecisionRecallCurve);

        return new ExperimentResult(name, avgAuc, avgPrAuc);
    }

    public (float[] Uncalibrated, float[] Platt, float[] Isotonic) CompareCalibration(
        IDataView trainFit,
        IDataView calib,
        IDataView validation)
    {
        // 1. Train base model on trainFit
        var basePipeline = BuildPipeline();
        var baseModel = basePipeline.Fit(trainFit);

        // 2. Score calib slice for fitting calibrators
        var calibScored = baseModel.Transform(calib);

        // 3. Fit Platt and Isotonic calibrators
        var plattEstimator = _ml.BinaryClassification.Calibrators.Platt(labelColumnName: "Label", scoreColumnName: "Score");
        var isotonicEstimator = _ml.BinaryClassification.Calibrators.Isotonic(labelColumnName: "Label", scoreColumnName: "Score");

        var plattModel = plattEstimator.Fit(calibScored);
        var isotonicModel = isotonicEstimator.Fit(calibScored);

        // 4. Score validation with base model
        var valBaseScored = baseModel.Transform(validation);
        var uncalibratedProbs = valBaseScored.GetColumn<float>("Probability").ToArray();

        // 5. Transform scored validation through each calibrator
        var valPlattScored = plattModel.Transform(valBaseScored);
        var plattProbs = valPlattScored.GetColumn<float>("Probability").ToArray();

        var valIsotonicScored = isotonicModel.Transform(valBaseScored);
        var isotonicProbs = valIsotonicScored.GetColumn<float>("Probability").ToArray();

        return (uncalibratedProbs, plattProbs, isotonicProbs);
    }

    public static List<ReliabilityBin> ComputeReliability(float[] probs, bool[] labels, int nBins = 10)
    {
        var bins = new List<ReliabilityBin>();

        for (int b = 0; b < nBins; b++)
        {
            var matchingIndices = Enumerable.Range(0, probs.Length)
                .Where(i =>
                {
                    int binIndex = (int)(probs[i] * nBins);
                    binIndex = Math.Min(binIndex, nBins - 1);
                    return binIndex == b;
                })
                .ToList();

            if (matchingIndices.Count == 0)
                continue;

            int count = matchingIndices.Count;
            double meanPredicted = matchingIndices.Average(i => probs[i]);
            double observed = matchingIndices.Count(i => labels[i]) / (double)count;
            double gap = meanPredicted - observed; // signed gap: positive = overconfident, negative = underconfident

            bins.Add(new ReliabilityBin(b, count, meanPredicted, observed, gap));
        }

        return bins;
    }

    public static double BrierScore(float[] probs, bool[] labels)
    {
        return probs.Zip(labels, (p, l) =>
        {
            double actual = l ? 1.0 : 0.0;
            double diff = p - actual;
            return diff * diff;
        }).Average();
    }

    public static double ExpectedCalibrationError(List<ReliabilityBin> bins, int totalCount)
    {
        return bins.Sum(b => Math.Abs(b.Gap) * b.Count) / totalCount;
    }
}