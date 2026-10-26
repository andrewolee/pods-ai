// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace DatasetManager
{
    public abstract class SamplesPageBase : Page, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected readonly DatasetRepository Repository;
        protected abstract string WavFolderPath { get; }
        protected abstract LogViewerControl LogViewer { get; }
        protected abstract SamplesGridControl SamplesGrid { get; }
        public string Category { get; }

        protected void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService?.GoBack();
        }

        /// <summary>
        /// Runs a Python script asynchronously with the specified
        /// arguments and captures its output.
        /// </summary>
        /// <param name="scriptPath">The path to the Python script to run.</param>
        /// <param name="args">The arguments to pass to the Python script.</param>
        /// <returns>The captured output from the Python script.</returns>
        protected async Task<string> RunPythonAsync(string scriptPath, string args)
        {
            try
            {
                SamplesGrid.InferenceEnabled = false;

                // Clear textbox.
                LogViewer.LogTextBox.Text = string.Empty;

                string[] commandLineArgs = Environment.GetCommandLineArgs();
                string rootPath = @".";
                if (commandLineArgs.Length > 1)
                {
                    rootPath = commandLineArgs[1];
                }
                string fullArguments = $"\"{scriptPath}\" {args}";
                LogViewer.AppendInfo($"> python {fullArguments.Trim()}");

                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = fullArguments,
                    WorkingDirectory = rootPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                var process = new Process
                {
                    StartInfo = psi,
                    EnableRaisingEvents = true
                };

                var outputReaderCompleted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                var errorReaderCompleted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

                process.OutputDataReceived += (s, e) =>
                {
                    if (e.Data == null)
                    {
                        outputReaderCompleted.TrySetResult();
                    }
                    else
                    {
                        Dispatcher.Invoke(() =>
                        {
                            LogViewer.AppendInfo(e.Data);
                        });
                    }
                };

                process.ErrorDataReceived += (s, e) =>
                {
                    if (e.Data == null)
                    {
                        errorReaderCompleted.TrySetResult();
                    }
                    else
                    {
                        Dispatcher.Invoke(() =>
                        {
                            LogViewer.AppendError(e.Data);
                        });
                    }
                };

                process.Start();

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await process.WaitForExitAsync();
                await Task.WhenAll(outputReaderCompleted.Task, errorReaderCompleted.Task);
            }
            finally
            {
                SamplesGrid.InferenceEnabled = true;
            }

            return LogViewer.LogTextBox.Text;
        }

        /// <summary>
        /// Finds tags in the output string that start with the specified
        /// prefix and extracts the confidence value if present.
        /// </summary>
        /// <param name="prefix">The prefix to search for in the output string.</param>
        /// <param name="output">The output string to search.</param>
        /// <param name="confidence">The extracted confidence value, if present.</param>
        /// <returns>The extracted tags string.</returns>
        private string FindTagsInOutput(string prefix, string output, out double confidence)
        {
            confidence = 0;

            string? line = output
                .Split('\n')
                .FirstOrDefault(l => l.StartsWith(prefix));
            if (line != null)
            {
                string remainder = line.Substring(prefix.Length);

                int paren = remainder.IndexOf('(');
                if (paren < 0)
                {
                    confidence = 1.0;
                    return remainder.Trim();
                }

                string tags = remainder.Substring(0, paren);
                string confidencePrefix = "confidence:";
                int start = remainder.IndexOf(confidencePrefix);
                int end = remainder.IndexOf(')', start);
                if (start >= 0 && end > start)
                {
                    string value = remainder
                        .Substring(start + confidencePrefix.Length,
                                   end - start - confidencePrefix.Length)
                        .Trim();

                    double.TryParse(value, out confidence);
                }
                return tags.Trim();
            }
            return string.Empty;
        }

        /// <summary>
        /// Gets the arguments used to run inference on the selected sample.
        /// </summary>
        protected virtual string GetInferenceArguments(SampleRecord sample)
        {
            return sample.GetWavFilePath(WavFolderPath);
        }

        /// <summary>
        /// Handles the InferenceRequested event from the SamplesGrid
        /// control.  It runs a Python script to perform inference on
        /// the selected sample and updates the sample's tags and
        /// confidence based on the output.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="sample">The sample record for which inference is requested.</param>
        protected async void SamplesGrid_InferenceRequested(object? sender, SampleRecord sample)
        {
            string result = await RunPythonAsync("src\\run_inference.py", GetInferenceArguments(sample));

            double confidence;
            string tag = FindTagsInOutput("Global prediction: ", result, out confidence);
            double dummyConfidence;
            string tags = FindTagsInOutput("Global predictions: ", result, out dummyConfidence);
            if (string.IsNullOrEmpty(tags))
            {
                tags = tag;
            }
            if (!string.IsNullOrEmpty(tags))
            {
                sample.InferredTags = tags;
                sample.InferredConfidence = confidence * 100.0;
            }
        }

        public bool IsNotResident => !string.Equals(Category, "resident", StringComparison.OrdinalIgnoreCase);

        public ICollectionView Samples { get; }

        protected SamplesPageBase(DatasetRepository repository, ObservableCollection<SampleRecord> samples, string category)
        {
            Repository = repository;
            Category = category;

            Samples = CollectionViewSource.GetDefaultView(samples);
            Samples.Filter = o => o is SampleRecord sample && sample.Category == category;

            DataContext = this;
        }
    }
}
