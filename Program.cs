using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using ReturnRisk.Core;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

Console.OutputEncoding = System.Text.Encoding.UTF8;

var mlContext = new MLContext(seed: 42);
var loader = new DataLoader(mlContext);
var runner = new Experiments(mlContext);

string dataPath = "known_transformed.csv";

// 1. Load both Honest (No CE) and Leaky (With CE) datasets
var dataNoCe = loader.Load(dataPath, includeTargetEncoded: false);
var (trainNoCe, _, testNoCe) = loader.SplitByTime(dataNoCe);

var dataWithCe = loader.Load(dataPath, includeTargetEncoded: true);
var (trainWithCe, _, testWithCe) = loader.SplitByTime(dataWithCe);

// 2. Run Test Evaluations (One single unbiased look)
var resultTestA = runner.RunTemporal("Config A (Honest, CE Dropped)", trainNoCe, testNoCe);
var resultTestB = runner.RunTemporal("Config B (Leaky, CE Kept)", trainWithCe, testWithCe);

Console.WriteLine("\n=======================================================");
Console.WriteLine("           FINAL UNTOUCHED TEST SET RESULTS            ");
Console.WriteLine("=======================================================");
Console.WriteLine($"{"Configuration",-32} | {"Test AUC",-10} | {"Test PR-AUC",-11}");
Console.WriteLine(new string('-', 59));
Console.WriteLine($"{resultTestA.Name,-32} | {resultTestA.Auc,10:F4} | {resultTestA.PrAuc,11:F4}");
Console.WriteLine($"{resultTestB.Name,-32} | {resultTestB.Auc,10:F4} | {resultTestB.PrAuc,11:F4}");

// 3. Compute Measured Precision and Recall at Cost-Optimal Tau for Config A
var modelA = runner.BuildPipeline().Fit(trainNoCe);
var testPredictionsA = modelA.Transform(testNoCe);

var probs = testPredictionsA.GetColumn<float>("Probability").ToArray();
var labels = testNoCe.GetColumn<bool>("Label").ToArray();

// Extract raw test prices to compute per-order tau
int priceIdx = loader.GetFeatureIndex(dataPath, "item_price", includeTargetEncoded: false);
var testRows = mlContext.Data.CreateEnumerable<OrderData>(testNoCe, reuseRowObject: false).ToArray();
var rawTestPrices = testRows.Select(r => (double)r.Features[priceIdx]).ToArray();

var profile = CostModel.MainstreamD2C;
var scaledPrices = CostModel.ScalePrices(rawTestPrices.Select(p => (float)p).ToArray(), profile);

int tp = 0, fp = 0, fn = 0, tn = 0;
for (int i = 0; i < labels.Length; i++)
{
    double tau = CostModel.Tau(scaledPrices[i], profile);
    bool predictedPositive = probs[i] > tau; // Intervene
    bool actualPositive = labels[i];         // Returned

    if (predictedPositive && actualPositive) tp++;
    else if (predictedPositive && !actualPositive) fp++;
    else if (!predictedPositive && actualPositive) fn++;
    else tn++;
}

double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
double interventionRate = (double)(tp + fp) / labels.Length;

Console.WriteLine($"\n--- Operational Metrics on Test ({profile.Name} @ Cost-Optimal Tau) ---");
Console.WriteLine($"Intervention Rate: {interventionRate * 100:F2}%");
Console.WriteLine($"Precision:         {precision:F4} ({tp} correct interventions / {tp + fp} total)");
Console.WriteLine($"Recall:            {recall:F4} ({tp} intercepted returns / {tp + fn} total returns)");

// 4. Save the trained production model (Config A)
string modelPath = Path.Combine(AppContext.BaseDirectory, "returnrisk-model.zip");
mlContext.Model.Save(modelA, trainNoCe.Schema, modelPath);

Console.WriteLine($"\n[Model Export] Model successfully saved to: {modelPath}");

// 5. Print first test row features and expected probability for Swagger testing
var firstOrder = testRows.First();
float firstProb = probs[0];
double firstPrice = scaledPrices[0];

Console.WriteLine("\n=======================================================");
Console.WriteLine("             SWAGGER TEST PAYLOAD DATA                 ");
Console.WriteLine("=======================================================");
Console.WriteLine($"Expected Model Probability: {firstProb:F4}");
Console.WriteLine($"Sample Price: {firstPrice:F2}");
Console.WriteLine("\nFeatures JSON Array (paste into 'features' field):");
Console.WriteLine("[" + string.Join(", ", firstOrder.Features.Select(f => f.ToString("G7"))) + "]");
Console.WriteLine("=======================================================");

// 6. Generate Reliability and Tau Charts
var bins = Experiments.ComputeReliability(probs, labels);
string plotDir = Path.Combine(AppContext.BaseDirectory, "plots");
Directory.CreateDirectory(plotDir);

string reliabilityPath = Path.Combine(plotDir, "reliability_curve.png");
Plotting.GenerateReliabilityCurve(bins, reliabilityPath);

string tauFull = Path.Combine(plotDir, "tau_vs_price_full.png");
string tauZoomed = Path.Combine(plotDir, "tau_vs_price_zoomed.png");
Plotting.GenerateTauVsPrice(tauFull, 20000.0);
Plotting.GenerateTauVsPrice(tauZoomed, 3000.0);

// 7. Latency Benchmarking Setup
var services = new ServiceCollection();
services.AddPredictionEnginePool<OrderScoringInput, OrderScoringOutput>()
    .FromFile("ReturnRisk", modelPath);
var serviceProvider = services.BuildServiceProvider();
var pool = serviceProvider.GetRequiredService<PredictionEnginePool<OrderScoringInput, OrderScoringOutput>>();

var benchmarkRequest = new ScoreRequest(
    Features: firstOrder.Features,
    Price: firstPrice,
    Merchant: CostModel.ValueFashion
);

// Match your API port from Swagger (e.g. 7163)
string apiBaseUrl = "https://localhost:7163";

Console.WriteLine("\nStarting latency benchmark against " + apiBaseUrl + " ...");
await LatencyBenchmark.Run(pool, apiBaseUrl, benchmarkRequest);
Console.WriteLine($"Stopwatch HighRes: {Stopwatch.IsHighResolution}, Freq: {Stopwatch.Frequency}");

Console.WriteLine("\nDone! Press any key to exit...");
Console.ReadKey();