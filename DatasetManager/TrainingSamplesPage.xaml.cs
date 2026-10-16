// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System.Windows;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TrainingSamplesPage.xaml
    /// </summary>
    public partial class TrainingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\wav";

        private string GetStartTimePSTStringForPastDays(int days)
        {
            TimeZoneInfo pacificZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
            DateTimeOffset pastDaysAgoPacific =
                TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, pacificZone)
                .AddDays(-days);
            return pastDaysAgoPacific.ToString("yyyy_MM_dd_HH_mm_ss") + "_PST";
        }

        protected async void MoreFalseNegatives_Click(object sender, RoutedEventArgs e)
        {
            if (!string.Equals(Category, "resident", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            if (Repository.ProposedTrainingSamples.Count(sample => sample.Category == "resident") == 0)
            {
                int days = OneWeekRadio.IsChecked == true ? 7 :
                           TwoWeekRadio.IsChecked == true ? 14 :
                           30;

                string timestamp = GetStartTimePSTStringForPastDays(days);
                string result = await RunPythonAsync(@"src\process_false_negatives.py", $"--start {timestamp} --end now --output-dir output/wav/resident");
                string? trainingCsvSection = DatasetRepository.ExtractCsvSection(
                    result,
                    "Proposed rows for output/csv/new_manual_training_samples.csv:");
                List<SampleRecord> newTrainingSamples = (trainingCsvSection != null)
                    ? DatasetRepository.LoadSamplesFromText(trainingCsvSection)
                    : new();

                newTrainingSamples.RemoveAll(sample =>
                    Repository.FindOverlapsIn(
                        Repository.TrainingSamples,
                        samplesSeconds: 3,
                        sample,
                        sampleSeconds: 3).Any());
                newTrainingSamples.RemoveAll(sample =>
                    Repository.FindOverlapsIn(
                        Repository.ProposedTrainingSamples,
                        samplesSeconds: 3,
                        sample,
                        sampleSeconds: 3).Any());
                newTrainingSamples.RemoveAll(sample =>
                    Repository.FindOverlapsIn(
                        Repository.RejectedTrainingSamples,
                        samplesSeconds: 3,
                        sample,
                        sampleSeconds: 3).Any());

                Repository.ProposeTrainingSamples(newTrainingSamples);
            }

            NavigationService?.Navigate(new AddTrainingSamplesPage(Repository, Category));
        }

        public TrainingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.TrainingSamples, category)
        {
            InitializeComponent();
        }
    }
}
