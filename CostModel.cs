using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReturnRisk.Core;

public record MerchantProfile(string Name, double MedianOrderValue, double MarginRate, double ShippingPerLeg, double Handling)
{
    public double ReturnCost => 2 * ShippingPerLeg + Handling;
}

public static class CostModel
{
    public static readonly MerchantProfile ValueFashion = new("Value Fashion", 499, 0.25, 60, 30);
    public static readonly MerchantProfile MainstreamD2C = new("Mainstream D2C", 1299, 0.40, 75, 40);
    public static readonly MerchantProfile PremiumApparel = new("Premium Apparel", 4500, 0.55, 110, 70);
    public static readonly MerchantProfile Electronics = new("Electronics", 15000, 0.08, 180, 150);

    public static double Tau(double itemPrice, MerchantProfile p)
    {
        double margin = itemPrice * p.MarginRate;
        return margin / (p.ReturnCost + margin);
    }

    public static double AverageProfit(
        bool[] labels,
        double[] prices,
        MerchantProfile profile,
        Func<int, bool> interveneDecision)
    {
        double totalProfit = 0.0;
        int n = labels.Length;

        for (int i = 0; i < n; i++)
        {
            if (interveneDecision(i))
                continue;

            if (labels[i])
            {
                totalProfit -= profile.ReturnCost;
            }
            else
            {
                totalProfit += prices[i] * profile.MarginRate;
            }
        }

        return totalProfit / n;
    }

    public static double[] ScalePrices(float[] rawPrices, MerchantProfile profile)
    {
        var sorted = rawPrices.OrderBy(p => p).ToArray();
        double medianRaw = sorted.Length % 2 == 0
            ? (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2.0
            : sorted[sorted.Length / 2];

        double scaleFactor = profile.MedianOrderValue / medianRaw;
        return rawPrices.Select(p => (double)p * scaleFactor).ToArray();
    }

    public static double InterventionRate(int total, Func<int, bool> interveneDecision)
    {
        int count = 0;
        for (int i = 0; i < total; i++)
        {
            if (interveneDecision(i)) count++;
        }
        return (double)count / total;
    }
}
