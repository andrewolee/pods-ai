// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using CsvHelper;
using System.ComponentModel;
using System.Globalization;
using System.Net;
using System.Runtime.CompilerServices;

namespace DatasetManager
{
    [CsvRecord]
    public partial class SampleRecord : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        public string Category { get; set; } = "";
        public string NodeName { get; set; } = "";
        public string StartTimestamp { get; set; } = "";
        public string URI { get; set; } = "";

        [CsvIgnore]
        public DateTime StartTimestampUtc
        {
            get
            {
                // The URI is in the format "https://live.orcasound.net/bouts/new/port-townsend?time=2026-07-08T18%3A37%3A31.000Z"
                // We can extract the timestamp from the URI and parse it as a DateTime.
                var uri = new Uri(URI);
                var timeValue = uri.Query
                    .TrimStart('?')
                    .Split('&', StringSplitOptions.RemoveEmptyEntries)
                    .Select(parameter => parameter.Split('=', 2))
                    .Where(parts => parts.Length == 2)
                    .Where(parts => string.Equals(
                        WebUtility.UrlDecode(parts[0]),
                        "time",
                        StringComparison.OrdinalIgnoreCase))
                    .Select(parts => WebUtility.UrlDecode(parts[1]))
                    .FirstOrDefault();
                if (string.IsNullOrEmpty(timeValue))
                {
                    throw new InvalidOperationException(
                    $"No time parameter found in URI '{URI}'.");
                }

                return DateTime.Parse(
                    timeValue,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal |
                    DateTimeStyles.AdjustToUniversal);
            }
        }
        public string Description { get; set; } = "";
        [CsvColumn("Notes")]
        public string Source { get; set; } = "";
        private decimal? _confidence = null;
        public decimal? Confidence {
            get => _confidence;
            set
            {
                if (_confidence == value)
                {
                    return;
                }

                _confidence = value;

                OnPropertyChanged();
                OnPropertyChanged(nameof(ConfidenceRatio));
            }
        }
        private double? _inferredConfidence = null;
        [CsvIgnore]
        public double? InferredConfidence
        {
            get => _inferredConfidence;
            set
            {
                if (_inferredConfidence == value)
                {
                    return;
                }

                _inferredConfidence = value;

                OnPropertyChanged();
                OnPropertyChanged(nameof(InferredConfidenceRatio));
            }
        }

        [CsvIgnore]
        public double? ConfidenceRatio => Confidence == null ? null : ((double)Confidence) / 100.0;
        [CsvIgnore]
        public double? InferredConfidenceRatio => InferredConfidence == null ? null : InferredConfidence / 100.0;
        private string _moderatedTags = "";

        /// <summary>
        /// Gets or sets the moderated tags for the sample. This property
        /// is optional and can be empty.
        /// </summary>
        [CsvColumn("Tags", Optional = true)]
        public string ModeratedTags
        {
            get => _moderatedTags;
            set
            {
                if (_moderatedTags == value)
                {
                    return;
                }

                _moderatedTags = value;

                OnPropertyChanged();
            }
        }

        private string _inferredTags = "";

        /// <summary>
        /// Gets or sets the tags for the sample inferred by the current PODS-AI model.
        /// This property is ignored during CSV serialization.
        /// </summary>
        [CsvIgnore]
        public string InferredTags
        {
            get => _inferredTags;
            set
            {
                if (_inferredTags == value)
                {
                    return;
                }

                _inferredTags = value;

                OnPropertyChanged();
                OnPropertyChanged(nameof(HasInferredTags));
            }
        }

        /// <summary>
        /// Gets a value indicating whether the sample has inferred tags.
        /// </summary>
        [CsvIgnore]
        public bool HasInferredTags => !string.IsNullOrWhiteSpace(InferredTags);

        /// <summary>
        /// Raises the PropertyChanged event for the specified property name.
        /// </summary>
        /// <param name="propertyName">The name of the property that changed.</param>
        protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Construct the WAV file pat from a base directory, the category, node slug,
        /// and start timestamp.
        /// </summary>
        /// <param name="baseDirectory">Base filesystem directory</param>
        /// <returns></returns>
        public string GetWavFilePath(string baseDirectory)
        {
            string slug = NodeName.Replace('_', '-');
            string path = $"{baseDirectory}\\{Category}\\{slug}_{StartTimestamp}.wav";
            return path;
        }
    }
}
