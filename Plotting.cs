using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ScottPlot;

namespace ReturnRisk.Core;

public static class Plotting
{
    public static void GenerateReliabilityCurve(List<ReliabilityBin> bins, string outputPath)
    {
        var plot = new Plot();

        // 1. Diagonal reference line (perfect calibration y = x)
        var diag = plot.Add.Line(0, 0, 1, 1);
        diag.Color = Colors.Gray.WithAlpha(0.6);
        diag.LinePattern = LinePattern.Dashed;
        diag.LineWidth = 2;
        diag.LegendText = "Perfect Calibration";

        // 2. Extract bin coordinates
        var validBins = bins.Where(b => b.Count > 0).ToList();
        double[] xs = validBins.Select(b => b.MeanPredicted).ToArray();
        double[] ys = validBins.Select(b => b.Observed).ToArray();

        // Connect empirical bins with an accent line
        var line = plot.Add.ScatterLine(xs, ys);
        line.Color = Colors.DarkCyan;
        line.LineWidth = 2;
        line.LegendText = "Model Reliability";

        // 3. Add scatter markers sized proportionally to sample count
        double maxCount = validBins.Max(b => b.Count);
        for (int i = 0; i < validBins.Count; i++)
        {
            var marker = plot.Add.Marker(xs[i], ys[i]);
            marker.Shape = MarkerShape.FilledCircle;
            marker.Color = Colors.DarkCyan;

            float size = (float)(4 + 14 * Math.Sqrt(validBins[i].Count / maxCount));
            marker.Size = size;
        }

        // 4. Formatting and styling
        plot.Title("Reliability Curve (Test Set Calibration)");
        plot.Axes.Bottom.Label.Text = "Mean Predicted Probability";
        plot.Axes.Left.Label.Text = "Observed Return Rate";

        plot.Axes.SetLimits(0, 1, 0, 1);
        plot.ShowLegend(Edge.Bottom);

        plot.SavePng(outputPath, 800, 600);
        Console.WriteLine($"[Plot] Reliability curve saved to: {outputPath}");
    }

    public static void GenerateTauVsPrice(string outputPath, double maxPrice = 20000.0)
    {
        var plot = new Plot();

        int points = 300;
        double minPrice = 200.0;

        double[] prices = new double[points];
        for (int i = 0; i < points; i++)
        {
            prices[i] = minPrice + i * (maxPrice - minPrice) / (points - 1);
        }

        // Four profiles to plot
        var profiles = new (MerchantProfile Profile, Color LineColor)[]
        {
            (CostModel.ValueFashion, Colors.SteelBlue),
            (CostModel.MainstreamD2C, Colors.Teal),
            (CostModel.PremiumApparel, Colors.Orange),
            (CostModel.Electronics, Colors.Crimson)
        };

        foreach (var (profile, color) in profiles)
        {
            double[] taus = prices.Select(p => CostModel.Tau(p, profile)).ToArray();
            var line = plot.Add.ScatterLine(prices, taus);
            line.LineWidth = 2.5f;
            line.Color = color;
            line.LegendText = $"{profile.Name} ({profile.MarginRate * 100:F0}% margin)";

            // Highlight median operating point using MedianOrderValue
            if (profile.MedianOrderValue >= minPrice && profile.MedianOrderValue <= maxPrice)
            {
                double operatingTau = CostModel.Tau(profile.MedianOrderValue, profile);

                // Add operating point marker
                var dot = plot.Add.Marker(profile.MedianOrderValue, operatingTau);
                dot.Shape = MarkerShape.FilledCircle;
                dot.Size = 12;
                dot.Color = color;

                // Add dark inner ring for high contrast
                var innerDot = plot.Add.Marker(profile.MedianOrderValue, operatingTau);
                innerDot.Shape = MarkerShape.OpenCircle;
                innerDot.Size = 12;
                innerDot.Color = Colors.Black;
            }
        }

        plot.Title($"Cost-Optimal Decision Threshold (τ) vs. Item Price (Up to ₹{maxPrice:N0})");
        plot.Axes.Bottom.Label.Text = "Item Price (₹)";
        plot.Axes.Left.Label.Text = "Intervention Threshold (τ)";

        plot.Axes.SetLimits(minPrice, maxPrice, 0.0, 1.0);
        plot.ShowLegend(Edge.Bottom);

        plot.SavePng(outputPath, 800, 600);
        Console.WriteLine($"[Plot] Tau vs Price curve saved to: {outputPath}");
    }
}