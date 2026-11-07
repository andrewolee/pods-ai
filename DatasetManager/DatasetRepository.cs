// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using CsvHelper;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;

namespace DatasetManager
{
    public class StatsRow
    {
        public string Category { get; set; } = "";
        public int TrainingSampleCount { get; set; }
        public int TestingSampleCount { get; set; }
        public int TrainingWavCount { get; set; }
        public int TestingWavCount { get; set; }
        public int TestCorrectCount { get; set; }
        public int TestTotalCount { get; set; }
        public int TestFalsePositiveCount { get; set; }
        public int TestFalseNegativeCount { get; set; }

        public double TestCorrectRatio => (TestTotalCount > 0) ? 1.0 * TestCorrectCount / TestTotalCount : 0;
        public double TestFalsePositiveRatio => (TestTotalCount > 0) ? 1.0 * TestFalsePositiveCount / TestTotalCount : 0;
        public double TestFalseNegativeRatio => (TestTotalCount > 0) ? 1.0 * TestFalseNegativeCount / TestTotalCount : 0;
    }

    public class DatasetRepository
    {
        public ObservableCollection<SampleRecord> TrainingSamples = new();
        public ObservableCollection<SampleRecord> TestingSamples = new();
        public ObservableCollection<SampleRecord> ProposedTrainingSamples = new();
        public ObservableCollection<SampleRecord> ProposedTestingSamples = new();
        public ObservableCollection<SampleRecord> RejectedTrainingSamples = new();
        public ObservableCollection<SampleRecord> RejectedTestingSamples = new();
        private List<string> _trainingWavs = new();
        private List<string> _testingWavs = new();
        public ObservableCollection<StatsRow> Stats { get; } = new ObservableCollection<StatsRow>();

        /// <summary>
        /// Loads sample records from a CsvReader instance and returns them
        /// as a list of SampleRecord objects.
        /// </summary>
        /// <param name="csv">The CsvReader instance containing the sample records.</param>
        /// <returns>A list of sample records loaded from the CsvReader instance.</returns>
        private static List<SampleRecord> LoadSamplesFromCsv(CsvReader csv)
        {
            var records = csv.GetRecords<SampleRecord>();
            return records.ToList();
        }

        /// <summary>
        /// Loads sample records from a CSV-formatted string.
        /// </summary>
        /// <param name="csvText">The CSV-formatted string containing the sample records.</param>
        /// <returns>A list of sample records loaded from the CSV string.</returns>
        public static List<SampleRecord> LoadSamplesFromText(string csvText)
        {
            using var csv = CsvReader.FromText(
                csvText,
                new CsvReaderOptions
                {
                    FormatProvider = CultureInfo.InvariantCulture,
                });
            return LoadSamplesFromCsv(csv);
        }

        /// <summary>
        /// Loads sample records from a CSV file at the specified path.
        /// </summary>
        /// <param name="path">The path to the CSV file.</param>
        /// <returns>A list of sample records loaded from the CSV file.</returns>
        private static List<SampleRecord> LoadSamples(string path)
        {
            using var csv = CsvReader.FromFile(
                path,
                new CsvReaderOptions
                {
                    FormatProvider = CultureInfo.InvariantCulture,
                },
                FileOptions.None);
            return LoadSamplesFromCsv(csv);
        }

        /// <summary>
        /// Saves the provided sample records to a CSV file at the specified path.
        /// </summary>
        /// <param name="samples">The sample records to save.</param>
        /// <param name="path">The path to the CSV file.</param>
        private void SaveSamples(IEnumerable<SampleRecord> samples, string path)
        {
            using var csv = CsvWriter.ToFile(
                path,
                new CsvWriterOptions
                {
                    FormatProvider = CultureInfo.InvariantCulture,
                },
                FileOptions.None);
            csv.WriteRecords(samples);
        }

        /// <summary>
        /// Extracts a section of CSV text from the given text based on the specified header.
        /// </summary>
        /// <param name="text">The text containing the CSV data.</param>
        /// <param name="header">The header indicating the start of the CSV section.</param>
        /// <returns>The extracted CSV section, or null if the header is not found.</returns>
        public static string? ExtractCsvSection(string text, string header)
        {
            string marker = header + Environment.NewLine;

            int start = text.IndexOf(marker, StringComparison.Ordinal);
            if (start < 0)
            {
                return null;
            }

            start += marker.Length;

            int end = text.IndexOf(
                Environment.NewLine + Environment.NewLine,
                start,
                StringComparison.Ordinal);
            if (end < 0)
            {
                end = text.Length;
            }

            return text[start..end].Trim();
        }

        /// <summary>
        /// Get the root filesystem path for dataset files.
        /// </summary>
        /// <returns>The root path</returns>
        private string GetRootPath()
        {
            string[] args = Environment.GetCommandLineArgs();
            string rootPath = @".";
            if (args.Length > 1)
            {
                rootPath = args[1];
            }
            return rootPath;
        }

        /// <summary>
        /// Reads the model comparison file and processes the confusion matrix.
        /// </summary>
        private void ReadModelComparison()
        {
            string rootPath = GetRootPath();
            string modelComparisonPath = Path.Combine(rootPath, "model-comparison-podsai.txt");
            if (File.Exists(modelComparisonPath))
            {
                string text = File.ReadAllText(modelComparisonPath);
                string[] lines = text.Split('\n');

                // Find the line "Confusion Matrix for podsai (rows=actual, cols=predicted):"
                int headerIndex = Array.FindIndex(lines, l => l.StartsWith("Confusion Matrix for podsai"));
                if (headerIndex < 0 || headerIndex + 1 >= lines.Length)
                {
                    return;
                }

                // Parse headers so we can the labels and offsets for each column.
                string[] headers = lines[headerIndex + 1]
                    .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                int totalColumn = Array.IndexOf(headers, "total");
                for (int i = headerIndex + 2; i < lines.Length; i++)
                {
                    string line = lines[i];
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        break;
                    }

                    string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2)
                    {
                        continue;
                    }
                    string category = parts[0];
                    int column = Array.IndexOf(headers, category);
                    int correct = column >= 0 ? int.Parse(parts[column + 1]) : 0;
                    int falsePositives = 0;
                    int falseNegatives = 0;
                    int total = int.Parse(parts[^1]); // last value
                    if (category == "resident" || category == "transient" || category == "humpback")
                    {
                        falseNegatives = total - correct;
                    }
                    else
                    {
                        int residentColumn = Array.IndexOf(headers, "resident");
                        int residents = (category != "resident" && residentColumn >= 0) ? int.Parse(parts[1 + residentColumn]) : 0;
                        int transientColumn = Array.IndexOf(headers, "transient");
                        int transients = (category != "transient" && transientColumn >= 0) ? int.Parse(parts[1 + transientColumn]) : 0;
                        int humpbackColumn = Array.IndexOf(headers, "humpback");
                        int humpbacks = (category != "humpback" && humpbackColumn >= 0) ? int.Parse(parts[1 + humpbackColumn]) : 0;
                        falsePositives = residents + transients + humpbacks;
                    }

                    StatsRow? row = Stats.FirstOrDefault(s => s.Category == category);
                    if (row != null)
                    {
                        row.TestCorrectCount = correct;
                        row.TestTotalCount = total;
                        row.TestFalsePositiveCount = falsePositives;
                        row.TestFalseNegativeCount = falseNegatives;
                    }
                    else
                    {
                        Stats.Add(new StatsRow
                        {
                            Category = category,
                            TestCorrectCount = correct,
                            TestTotalCount = total,
                            TestFalsePositiveCount = falsePositives,
                            TestFalseNegativeCount = falseNegatives,
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Populates the Stats collection with statistics for each category
        /// based on the provided training and testing samples and WAV files.
        /// </summary>
        /// <param name="trainingSamples">The collection of training samples.</param>
        /// <param name="testingSamples">The collection of testing samples.</param>
        /// <param name="trainingWavs">The list of training WAV file paths.</param>
        /// <param name="testingWavs">The list of testing WAV file paths.</param>
        private void PopulateStats(
            ObservableCollection<SampleRecord> trainingSamples,
            ObservableCollection<SampleRecord> testingSamples,
            List<string> trainingWavs,
            List<string> testingWavs)
        {
            // Recompute stats from scratch so callers can refresh after collection changes.
            Stats.Clear();

            var categories =
                trainingSamples.Select(s => s.Category)
                .Union(testingSamples.Select(s => s.Category))
                .OrderBy(c => c);

            foreach (var category in categories)
            {

                Stats.Add(new StatsRow
                {
                    Category = category,
                    TrainingSampleCount =
                        trainingSamples.Count(s => s.Category == category),

                    TestingSampleCount =
                        testingSamples.Count(s => s.Category == category),

                    TrainingWavCount =
                        trainingWavs.Count(s => s.Contains(category)),

                    TestingWavCount =
                        testingWavs.Count(s => s.Contains(category)),
                });
            }
        }

        /// <summary>
        /// Loads the dataset repository from the specified root path,
        /// reading training and testing samples from CSV files and
        /// populating the corresponding collections. If the CSV files
        /// cannot be loaded, an error message is displayed, and the
        /// application exits.
        /// </summary>
        /// <returns>The loaded DatasetRepository instance.</returns>
        public static DatasetRepository Load()
        {
            var repository = new DatasetRepository();

            string rootPath = repository.GetRootPath();

            try
            {
                List<SampleRecord> samples = LoadSamples(Path.Combine(rootPath, "output", "csv", "training_3s_samples.csv"));
                repository.TrainingSamples = new ObservableCollection<SampleRecord>(samples);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not load output\\csv\\training_3s_samples.csv. Please ensure the CSV file exists and is accessible, and either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            try
            {
                List<SampleRecord> samples =
                    LoadSamples(Path.Combine(rootPath, "output", "csv", "testing_60s_samples.csv"));
                repository.TestingSamples = new ObservableCollection<SampleRecord>(samples);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not load testing samples. Please ensure the CSV file exists and is accessible, and either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            repository._trainingWavs = Directory.Exists(Path.Combine(rootPath, "output", "wav"))
                ? Directory.GetFiles(Path.Combine(rootPath, "output", "wav"), "*.wav", SearchOption.AllDirectories).ToList()
                : new List<string>();

            string testingWavPath = Path.Combine(rootPath, "output", "testing-wav");
            repository._testingWavs = Directory.Exists(testingWavPath)
                ? Directory.GetFiles(testingWavPath, "*.wav", SearchOption.AllDirectories).ToList()
                : new List<string>();

            repository.PopulateStats(repository.TrainingSamples, repository.TestingSamples, repository._trainingWavs, repository._testingWavs);

            repository.ReadModelComparison();

            return repository;
        }

        /// <summary>
        /// Saves the training and testing samples to their respective CSV files.
        /// </summary>
        public void Save()
        {
            string rootPath = GetRootPath();
            try
            {
                SaveSamples(TrainingSamples, Path.Combine(rootPath, "output", "csv", "training_3s_samples.csv"));
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Could not save output\\csv\\training_3s_samples.csv. Either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
                Environment.Exit(1);
            }

            try
            {
                SaveSamples(TestingSamples, Path.Combine(rootPath, "output", "csv", "testing_60s_samples.csv"));
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Could not save testing samples. Either run this application from the pods-ai directory, or specify the path to it on the command line.\n\nError: {ex.Message}",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
                Environment.Exit(1);
            }
        }

        /// <summary>
        /// Proposes new training samples by adding them to the ProposedTrainingSamples collection.
        /// </summary>
        /// <param name="newTrainingSamples">The list of new training samples to propose.</param>
        public void ProposeTrainingSamples(List<SampleRecord> newTrainingSamples)
        {
            foreach (var sample in newTrainingSamples)
            {
                ProposedTrainingSamples.Add(sample);
            }
        }

        /// <summary>
        /// Proposes new testing samples by adding them to the ProposedTestingSamples collection.
        /// </summary>
        /// <param name="newTestingSamples">The list of new testing samples to propose.</param>
        public void ProposeTestingSamples(List<SampleRecord> newTestingSamples)
        {
            foreach (var sample in newTestingSamples)
            {
                ProposedTestingSamples.Add(sample);
            }
        }

        /// <summary>
        /// Accepts a testing sample by adding it to the TestingSamples collection.
        /// </summary>
        /// <param name="testingSample">The testing sample to accept.</param>
        public void AcceptTestingSample(SampleRecord testingSample)
        {
            int index = 0;
            while (index < TestingSamples.Count && CompareSamples(TestingSamples[index], testingSample) < 0)
            {
                index++;
            }
            TestingSamples.Insert(index, testingSample);
            ProposedTestingSamples.Remove(testingSample);
        }

        /// <summary>
        /// Rejects a testing sample by adding it to the RejectedTestingSamples collection.
        /// </summary>
        /// <param name="testingSample">The testing sample to reject.</param>
        public void RejectTestingSample(SampleRecord testingSample)
        {
            RejectedTestingSamples.Add(testingSample);
            ProposedTestingSamples.Remove(testingSample);
        }

        /// <summary>
        /// Compares two SampleRecord objects based on their Category, NodeName, and StartTimestampUtc properties.
        /// </summary>
        /// <param name="left">The first SampleRecord to compare.</param>
        /// <param name="right">The second SampleRecord to compare.</param>
        /// <returns>An integer that indicates the relative order of the objects being compared.</returns>
        private static int CompareSamples(SampleRecord left, SampleRecord right)
        {
            int result = string.Compare(left.Category, right.Category, StringComparison.OrdinalIgnoreCase);
            if (result != 0)
            {
                return result;
            }

            result = string.Compare(left.NodeName, right.NodeName, StringComparison.OrdinalIgnoreCase);
            if (result != 0)
            {
                return result;
            }

            return DateTime.Compare(left.StartTimestampUtc, right.StartTimestampUtc);
        }

        /// <summary>
        /// Finds overlapping samples in the given list of samples based on the specified sample and time window in seconds.
        /// </summary>
        /// <param name="samples">The list of samples to search for overlaps.</param>
        /// <param name="samplesSeconds">The duration of the samples to find overlaps for.</param>
        /// <param name="sample">The sample to find overlaps for.</param>
        /// <param name="sampleSeconds">The duration of the sample to find overlaps for.</param>
        /// <returns>A list of overlapping samples.</returns>
        public List<SampleRecord> FindOverlapsIn(IEnumerable<SampleRecord> samples, int samplesSeconds, SampleRecord sample, int sampleSeconds)
        {
            DateTime sampleStart = sample.StartTimestampUtc;
            DateTime sampleEnd = sampleStart.AddSeconds(sampleSeconds);

            return samples.Where(s =>
            {
                if (s.NodeName != sample.NodeName)
                {
                    return false;
                }

                DateTime otherStart = s.StartTimestampUtc;
                DateTime otherEnd = otherStart.AddSeconds(samplesSeconds);

                return sampleStart < otherEnd && otherStart < sampleEnd;
            }).ToList();
        }

        /// <summary>
        /// Accepts a training sample by adding it to the TrainingSamples collection
        /// and removing it from the ProposedTrainingSamples collection.
        /// </summary>
        /// <param name="trainingSample">The training sample to accept.</param>
        public void AcceptTrainingSample(SampleRecord trainingSample)
        {
            int index = 0;
            while (index < TrainingSamples.Count && CompareSamples(TrainingSamples[index], trainingSample) < 0)
            {
                index++;
            }
            TrainingSamples.Insert(index, trainingSample);
            ProposedTrainingSamples.Remove(trainingSample);

            // Remove any overlapping samples from testing samples.
            var overlaps = FindOverlapsIn(
                TestingSamples,
                samplesSeconds: 60,
                sample: trainingSample,
                sampleSeconds: 3);
            foreach (var overlap in overlaps)
            {
                TestingSamples.Remove(overlap);
            }

            // Refresh stats so UI shows updated counts after accepting a sample
            PopulateStats(TrainingSamples, TestingSamples, _trainingWavs, _testingWavs);
        }

        /// <summary>
        /// Rejects a training sample by adding it to the RejectedTrainingSamples
        /// collection and removing it from the ProposedTrainingSamples collection.
        /// </summary>
        /// <param name="trainingSample">The training sample to reject.</param>
        public void RejectTrainingSample(SampleRecord trainingSample)
        {
            RejectedTrainingSamples.Add(trainingSample);
            ProposedTrainingSamples.Remove(trainingSample);
        }
    }
}
