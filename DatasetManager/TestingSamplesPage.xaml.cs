// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System.Windows;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TestingSamplesPage.xaml
    /// </summary>
    public partial class TestingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\testing-wav";

        /// <summary>
        /// Gets the count of false positives for the current category
        /// from the repository's statistics.
        /// </summary>
        public int FalsePositiveCount
        {
            get
            {
                StatsRow? row = Repository.Stats.FirstOrDefault(s => s.Category == Category);
                return (row != null) ? row.TestFalsePositiveCount : 0;
            }
        }

        public bool HasFalsePositives => FalsePositiveCount > 0;

        public TestingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.TestingSamples, category)
        {
            InitializeComponent();
        }

        /// <summary>
        /// Gets the start time in Pacific Standard Time (PST) for the past week
        /// and returns it as a formatted string.
        /// </summary>
        /// <returns>A formatted string representing the start time in PST a given number of days ago.</returns>
        private string GetStartTimePSTStringForPastDays(int days)
        {
            TimeZoneInfo pacificZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
            DateTimeOffset pastDaysAgoPacific =
                TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, pacificZone)
                .AddDays(-days);
            return pastDaysAgoPacific.ToString("yyyy_MM_dd_HH_mm_ss") + "_PST";
        }

        /// <summary>
        /// Proposes new testing samples from the given result string and header.
        /// It extracts the CSV section, loads the samples, and removes any duplicates
        /// that overlap with existing training, proposed, or rejected samples.
        /// </summary>
        /// <param name="result">The result string containing CSV sections.</param>
        /// <param name="header">The header of the CSV section to extract.</param>
        private void ProposeSamplesFromText(string result, string header)
        {
            string? testingCsvSection = DatasetRepository.ExtractCsvSection(result, header);

            List <SampleRecord> newTestingSamples = (testingCsvSection != null)
                   ? DatasetRepository.LoadSamplesFromText(testingCsvSection)
                   : new();

            // Remove any testing samples that overlap any samples already in the training set to avoid duplicates.
            newTestingSamples.RemoveAll(sample =>
                Repository.FindOverlapsIn(
                    Repository.TrainingSamples,
                    samplesSeconds: 3,
                    sample,
                    sampleSeconds: 60).Any());

            // Remove any samples that overlap any samples already the testing set to avoid duplicates.
            newTestingSamples.RemoveAll(sample =>
                Repository.FindOverlapsIn(
                    Repository.TestingSamples,
                    samplesSeconds: 60,
                    sample,
                    sampleSeconds: 60).Any());

            // Now remove any that overlap with any samples already proposed to avoid duplicates.
            newTestingSamples.RemoveAll(sample =>
                Repository.FindOverlapsIn(
                    Repository.ProposedTestingSamples,
                    samplesSeconds: 60,
                    sample,
                    sampleSeconds: 60).Any());

            // Finally, remove any that overlap with any samples already in the rejected set.
            newTestingSamples.RemoveAll(sample =>
                Repository.FindOverlapsIn(
                    Repository.RejectedTestingSamples,
                    samplesSeconds: 60,
                    sample,
                    sampleSeconds: 60).Any());

            Repository.ProposeTestingSamples(newTestingSamples);
        }

        /// <summary>
        /// Handles the click event for the "More False Positives" button.  If
        /// there are no proposed testing samples for the current category, it
        /// runs a Python script to process false positives and proposes new
        /// testing samples.  Finally, it navigates to the AddTestingSamplesPage.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The event data.</param>
        protected async void MoreFalsePositives_Click(object sender, RoutedEventArgs e)
        {
            if (Repository.ProposedTestingSamples.Count(sample => sample.Category == Category) == 0)
            {
                int days = OneWeekRadio.IsChecked == true ? 7 :
                           TwoWeekRadio.IsChecked == true ? 14 :
                           30;

                string timestamp = GetStartTimePSTStringForPastDays(days);

                string result = await RunPythonAsync(@"src\process_false_positives.py", $"--set testing --start {timestamp} --end now --category {Category}");

                ProposeSamplesFromText(result, $"Proposed rows for output/csv/testing_60s_samples.csv:");
            }

            NavigationService?.Navigate(new AddTestingSamplesPage(Repository, Category));
        }

        /// <summary>
        /// Handles the click event for the "Find more false negatives" button.
        /// It runs the false-negative processor for the testing set, proposes
        /// non-overlapping samples, and navigates to the AddTestingSamplesPage.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The event data.</param>
        protected async void MoreFalseNegatives_Click(object sender, RoutedEventArgs e)
        {
            if (!string.Equals(Category, "resident", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            if (Repository.ProposedTestingSamples.Count(sample => sample.Category == "resident") == 0)
            {
                int days = OneWeekRadio.IsChecked == true ? 7 :
                           TwoWeekRadio.IsChecked == true ? 14 :
                           30;

                string timestamp = GetStartTimePSTStringForPastDays(days);
                string result = await RunPythonAsync(
                    @"src\process_false_negatives.py",
                    $"--set testing --start {timestamp} --end now");

                ProposeSamplesFromText(
                    result,
                    "Proposed rows for output/csv/testing_60s_samples.csv:");
            }

            NavigationService?.Navigate(new AddTestingSamplesPage(Repository, Category));
        }

        /// <summary>
        /// Handles the click event for the "Find Mispredictions" button.
        /// If there are no proposed training samples for the current
        /// category, it runs a Python script to process mispredictions
        /// and proposes new training samples. Finally, it navigates to
        /// the AddTrainingSamplesPage.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The event data.</param>
        protected async void FindMispredictions_Click(object sender, RoutedEventArgs e)
        {
            if (Repository.ProposedTrainingSamples.Count(sample => sample.Category == Category) == 0)
            {
                var result = await RunPythonAsync(@"src\process_testing_set_mispredictions.py", $"--category {Category}");

                string? trainingCsvSection = DatasetRepository.ExtractCsvSection(result, "Proposed rows for output/csv/training_3s_samples.csv:");
                List<SampleRecord> newTrainingSamples = (trainingCsvSection != null)
                    ? DatasetRepository.LoadSamplesFromText(trainingCsvSection)
                    : new();

                // Remove any samples that overlap any samples already in the training set to avoid duplicates.
                newTrainingSamples.RemoveAll(sample =>
                    Repository.FindOverlapsIn(
                        Repository.TrainingSamples,
                        samplesSeconds: 3,
                        sample,
                        sampleSeconds: 3).Any());

                // Now remove any that overlap with any samples already proposed to avoid duplicates.
                newTrainingSamples.RemoveAll(sample =>
                    Repository.FindOverlapsIn(
                        Repository.ProposedTrainingSamples,
                        samplesSeconds: 3,
                        sample,
                        sampleSeconds: 3).Any());

                // Finally, remove any that overlap with any samples already in the rejected set.
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

        private bool _enableUpdateTags = true;

        public bool EnableUpdateTags
        {
            get => _enableUpdateTags;
            private set
            {
                if (_enableUpdateTags == value)
                {
                    return;
                }

                _enableUpdateTags = value;
                OnPropertyChanged();
            }
        }

        protected async void UpdateModeratedTags_Click(object sender, RoutedEventArgs e)
        {
            // Read the samples from OrcaHello and update the samples in
            // the repository with the moderated tags from OrcaHello.
            IEnumerable<SampleRecord> filteredSamples = Repository.TestingSamples
                .Where(sample => sample.Category == Category);

            EnableUpdateTags = false;
            try
            {
                await OrcaHelloHelper.UpdateModeratedTagsAsync(filteredSamples);
            }
            finally
            {
                EnableUpdateTags = true;
            }
        }
    }
}
