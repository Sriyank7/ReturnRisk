using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ReturnRisk.Core;

public class DataLoader
{
    private readonly MLContext _ml;

    // Temporal boundary constants (Days 0-254 train, 255-309 val, 310-364 test)
    private const double TrainStartDay = 0;
    private const double ValStartDay = 255;
    private const double TestStartDay = 310;
    private const double TestEndDayExclusive = 365;
    private const double CalibStartDay = 204;

    // The 3 structural columns that must never be treated as training features
    public static readonly string[] NonFeatureColumns = new[]
    {
        "order_item_id",
        "ord_time_num",
        "return"
    };

    // The 10 actual target-encoded (leaky) columns in known_transformed.csv
    public static readonly string[] TargetEncodedColumns = new[]
    {
        "ce_color",
        "ce_item_rounded",
        "ce_size",
        "ce_price",
        "ce_brand_rounded",
        "ce_day_rounded",
        "colour_item_ce",
        "size_item_ce",
        "ret_chance_bins",
        "user_ce_bins"
    };

    public DataLoader(MLContext mlContext)
    {
        _ml = mlContext;
    }

    public static string[] ReadHeader(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"CSV file not found at: {filePath}");

        using var reader = new StreamReader(filePath);
        var headerLine = reader.ReadLine() ?? throw new InvalidOperationException("CSV is empty");
        return headerLine.Split(',').Select(h => h.Trim().Trim('"')).ToArray();
    }

    public static int[] SelectFeatureIndices(string[] allColumns, bool includeTargetEncoded)
    {
        var missing = TargetEncodedColumns
            .Where(c => !allColumns.Contains(c, StringComparer.OrdinalIgnoreCase))
            .ToList();

        if (missing.Any())
        {
            throw new ArgumentException($"Expected target encoded columns missing from header: {string.Join(", ", missing)}");
        }

        var rejectSet = new HashSet<string>(NonFeatureColumns, StringComparer.OrdinalIgnoreCase);

        if (!includeTargetEncoded)
        {
            rejectSet.UnionWith(TargetEncodedColumns);
        }

        return Enumerable.Range(0, allColumns.Length)
            .Where(i => !rejectSet.Contains(allColumns[i]))
            .ToArray();
    }

    public IDataView Load(string filePath, bool includeTargetEncoded = true)
    {
        var headers = ReadHeader(filePath);
        int dayIndex = Array.FindIndex(headers, h => h.Equals("ord_time_num", StringComparison.OrdinalIgnoreCase));
        int labelIndex = Array.FindIndex(headers, h => h.Equals("return", StringComparison.OrdinalIgnoreCase));

        if (dayIndex == -1) throw new InvalidOperationException("Column 'ord_time_num' not found in CSV header.");
        if (labelIndex == -1) throw new InvalidOperationException("Column 'return' not found in CSV header.");

        var featureIndices = SelectFeatureIndices(headers, includeTargetEncoded);

        var columns = new List<TextLoader.Column>
        {
            new TextLoader.Column("Label", DataKind.Boolean, labelIndex),
            new TextLoader.Column("Day", DataKind.Single, dayIndex),
            new TextLoader.Column("Features", DataKind.Single, featureIndices.Select(i => new TextLoader.Range(i)).ToArray())
        };

        var options = new TextLoader.Options
        {
            HasHeader = true,
            Separators = new[] { ',' },
            Columns = columns.ToArray()
        };

        return _ml.Data.CreateTextLoader(options).Load(filePath);
    }

    public (IDataView Train, IDataView Validation, IDataView Test) SplitByTime(IDataView data)
    {
        var train = _ml.Data.FilterRowsByColumn(data, "Day", lowerBound: TrainStartDay, upperBound: ValStartDay);
        var val = _ml.Data.FilterRowsByColumn(data, "Day", lowerBound: ValStartDay, upperBound: TestStartDay);
        var test = _ml.Data.FilterRowsByColumn(data, "Day", lowerBound: TestStartDay, upperBound: TestEndDayExclusive);

        return (train, val, test);
    }

    public void AssertSplitIsValid(
        IDataView original,
        IDataView train,
        IDataView validation,
        IDataView test)
    {
        long CountRows(IDataView view) => view.GetColumn<float>("Day").LongCount();

        long originalCount = CountRows(original);
        long trainCount = CountRows(train);
        long valCount = CountRows(validation);
        long testCount = CountRows(test);

        if (trainCount == 0 || valCount == 0 || testCount == 0)
        {
            throw new InvalidOperationException(
                $"One or more splits are empty! Train: {trainCount}, Validation: {valCount}, Test: {testCount}");
        }

        long splitSum = trainCount + valCount + testCount;
        if (splitSum != originalCount)
        {
            throw new InvalidOperationException(
                $"Row count mismatch! Original: {originalCount}, but Train ({trainCount}) + Val ({valCount}) + Test ({testCount}) = {splitSum}");
        }

        var trainDays = train.GetColumn<float>("Day");
        var valDays = validation.GetColumn<float>("Day");
        var testDays = test.GetColumn<float>("Day");

        float maxTrainDay = trainDays.Max();
        float minValDay = valDays.Min();
        float maxValDay = valDays.Max();
        float minTestDay = testDays.Min();

        if (maxTrainDay >= minValDay)
        {
            throw new InvalidOperationException(
                $"Temporal overlap between Train and Validation! Max Train Day ({maxTrainDay}) >= Min Val Day ({minValDay})");
        }

        if (maxValDay >= minTestDay)
        {
            throw new InvalidOperationException(
                $"Temporal overlap between Validation and Test! Max Val Day ({maxValDay}) >= Min Test Day ({minTestDay})");
        }

        Console.WriteLine($"[Split Verified] Train: {trainCount} | Validation: {valCount} | Test: {testCount} | Total: {originalCount}");
    }

    public (IDataView TrainFit, IDataView Calib) SplitTrainForCalibration(IDataView train)
    {
        var trainFit = _ml.Data.FilterRowsByColumn(train, "Day", lowerBound: TrainStartDay, upperBound: CalibStartDay);
        var calib = _ml.Data.FilterRowsByColumn(train, "Day", lowerBound: CalibStartDay, upperBound: ValStartDay);

        return (trainFit, calib);
    }

    public int GetFeatureIndex(string filePath, string featureName, bool includeTargetEncoded)
    {
        var headers = ReadHeader(filePath);
        var featureIndices = SelectFeatureIndices(headers, includeTargetEncoded);
        int targetCsvIndex = Array.FindIndex(headers, h => h.Equals(featureName, StringComparison.OrdinalIgnoreCase));

        if (targetCsvIndex == -1)
            throw new ArgumentException($"Column '{featureName}' not found in CSV header.");

        int vectorIndex = Array.IndexOf(featureIndices, targetCsvIndex);
        if (vectorIndex == -1)
            throw new ArgumentException($"Column '{featureName}' is excluded from the feature vector.");

        return vectorIndex;
    }

    public static List<IDataView> SplitIntoRounds(MLContext ml, IDataView data, int nRounds)
    {
        var days = data.GetColumn<float>("Day").ToArray();
        float minDay = days.Min();
        float maxDay = days.Max();
        float step = (maxDay - minDay) / (float)nRounds;

        var rounds = new List<IDataView>();

        for (int r = 0; r < nRounds; r++)
        {
            float start = minDay + r * step;
            float end = (r == nRounds - 1) ? float.PositiveInfinity : minDay + (r + 1) * step;

            var roundView = ml.Data.FilterRowsByColumn(data, "Day", lowerBound: start, upperBound: end);
            rounds.Add(roundView);
        }

        return rounds;
    }
}