using Microsoft.ML.Data;
using ReturnRisk.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReturnRisk.Core;

public enum Decision
{
    Allow,
    Intervene
}

public record ScoreRequest(
    float[] Features,
    double Price,
    MerchantProfile Merchant
);

public record ScoreResponse(
    double RiskScore,
    double Threshold,
    Decision Action,
    double ExpectedProfitAllow,
    double ExpectedProfitIntervene,
    MerchantProfile Merchant
);

public class OrderScoringInput
{
    [VectorType(62)]
    public float[] Features { get; set; } = Array.Empty<float>();
}

public class OrderScoringOutput
{
    [ColumnName("Probability")]
    public float Probability { get; set; }

    public float Score { get; set; }
}