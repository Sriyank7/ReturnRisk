using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.ML;

namespace ReturnRisk.Core;

public static class LatencyBenchmark
{
    public static async Task Run(
        PredictionEnginePool<OrderScoringInput, OrderScoringOutput> pool,
        string apiBaseUrl,
        ScoreRequest sampleRequest)
    {
        const int warmUpCount = 100;
        const int iterations = 1000;

        // ----------------------------------------------------
        // 1. In-Process Model & Decision Latency
        // ----------------------------------------------------
        var input = new OrderScoringInput { Features = sampleRequest.Features };

        // Warm-up
        for (int i = 0; i < warmUpCount; i++)
        {
            var pred = pool.Predict("ReturnRisk", input);
            double tau = CostModel.Tau(sampleRequest.Price, sampleRequest.Merchant);
            _ = pred.Probability > tau ? Decision.Intervene : Decision.Allow;
        }

        var inProcessLatencies = new double[iterations];
        var sw = new Stopwatch();

        for (int i = 0; i < iterations; i++)
        {
            sw.Restart();
            var pred = pool.Predict("ReturnRisk", input);
            double tau = CostModel.Tau(sampleRequest.Price, sampleRequest.Merchant);
            _ = pred.Probability > tau ? Decision.Intervene : Decision.Allow;
            sw.Stop();
            inProcessLatencies[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(inProcessLatencies);
        int idx50 = (int)Math.Ceiling(iterations * 0.50) - 1;
        int idx99 = (int)Math.Ceiling(iterations * 0.99) - 1;

        double inProcessP50 = inProcessLatencies[idx50];
        double inProcessP99 = inProcessLatencies[idx99];

        // ----------------------------------------------------
        // 2. Full HTTP Round-Trip Latency (incl. JSON parse)
        // ----------------------------------------------------
        using var client = new HttpClient { BaseAddress = new Uri(apiBaseUrl) };

        // Warm-up
        for (int i = 0; i < warmUpCount; i++)
        {
            var res = await client.PostAsJsonAsync("/score", sampleRequest);
            res.EnsureSuccessStatusCode();
            _ = await res.Content.ReadFromJsonAsync<ScoreResponse>();
        }

        var httpLatencies = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            sw.Restart();
            var res = await client.PostAsJsonAsync("/score", sampleRequest);
            _ = await res.Content.ReadFromJsonAsync<ScoreResponse>();
            sw.Stop();
            httpLatencies[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(httpLatencies);
        double httpP50 = httpLatencies[idx50];
        double httpP99 = httpLatencies[idx99];

        Console.WriteLine("\n=======================================================");
        Console.WriteLine("                LATENCY BENCHMARK RESULTS               ");
        Console.WriteLine("=======================================================");
        Console.WriteLine($"{"Tier",-30} | {"p50 (ms)",-10} | {"p99 (ms)",-10}");
        Console.WriteLine(new string('-', 56));
        Console.WriteLine($"{"In-Process (Model + Tau)",-30} | {inProcessP50,10:F3} | {inProcessP99,10:F3}");
        Console.WriteLine($"{"HTTP Round-Trip (incl. JSON)",-30} | {httpP50,10:F3} | {httpP99,10:F3}");
        Console.WriteLine("=======================================================\n");
    }
}
