using System;
using System.Diagnostics;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MNISTdotNet
{
    internal class Program
    {
        private const string default_train_csv_file = "mnist-train.csv";
        private const string default_test_csv_file = "mnist-test.csv";
        private const string default_ml_model_file = "ml-model.mld";

        private static int image_length = 0;

        private static void Main(string[] args)
        {
            // Queue: Convert and create the CSV file for ML.net?
            Console.WriteLine("Do you want to convert and create the CSV file for ML.net? If the file is already exist, type 'N' to skip, else type 'Y' to avoid errors.");
            if (Console.ReadKey().Key == ConsoleKey.Y)
            {
                MNISTDataConvertor convertor = new MNISTDataConvertor(1000, 600);
                convertor.ConvertAndSave();

                image_length = convertor.image_length;
            }
            else
            {
                image_length = 784;
            }

            // Check the CSV file existion
            if ((!File.Exists(default_train_csv_file)) || (!File.Exists(default_test_csv_file)))
            {
                Console.WriteLine("The train or test CSV file is unexist.");
                Console.ReadKey(intercept: false);
                return;
            }

            // Initialize the machine learning context
            MLContext context = new MLContext();

            // Queue: Re-train the machine learning model?
            Console.WriteLine("Do you want to re-train the machine learning model for ML.net? If you want to use model already exist, type 'N' to skip, else type 'Y' to train it.");
            if (Console.ReadKey().Key == ConsoleKey.Y)
            {
                // Try to train the machine learning model by train data
                TrainMNIST(context: context, train_csv_file: default_train_csv_file, test_csv_file: default_test_csv_file);
            }

            Console.ReadKey(intercept: false);
            return;
        }

        private static void TrainMNIST(MLContext context, string train_csv_file, string test_csv_file, string model_file = default_ml_model_file)
        {
            // Common data loading configuration
            // Create a dataview for training
            IDataView trainingDataView = context.Data.LoadFromTextFile(path: train_csv_file,
                columns: new[]
                {
                    // The text loader for MNIST CSV format
                    new TextLoader.Column(nameof(MNISTData.ImageVector), DataKind.Single, 0, image_length - 1),
                    new TextLoader.Column(nameof(MNISTData.Number), DataKind.Single, image_length),
                },
                hasHeader: false, separatorChar: ',');

            // Create a dataview for testing
            IDataView testingDataView = context.Data.LoadFromTextFile(path: test_csv_file,
                columns: new[]
                {
                    // The text loader for MNIST CSV format
                    new TextLoader.Column(nameof(MNISTData.ImageVector), DataKind.Single, 0, image_length - 1),
                    new TextLoader.Column(nameof(MNISTData.Number), DataKind.Single, image_length),
                },
                hasHeader: false, separatorChar: ',');

            // Common data process configuration with pipeline data transformations
            EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> dataProcessPipeline =
                context.Transforms.Conversion.MapValueToKey("Label", nameof(MNISTData.Number)).
                Append(context.Transforms.Concatenate("Features", nameof(MNISTData.ImageVector)).
                AppendCacheCheckpoint(context));

            // Set the training algorithm, then create and config the modelBuilder
            SdcaMultiClassTrainer trainer = context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features");
            EstimatorChain<KeyToValueMappingTransformer> trainingPipeline =
                dataProcessPipeline.Append(trainer).
                Append(context.Transforms.Conversion.MapKeyToValue(nameof(MNISTData.Number), "Label"));

            // Train the model fitting to the DataSet
            Stopwatch watch = Stopwatch.StartNew();
            Console.WriteLine("=============== Training the model ===============");

            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"***** Training time: {elapsedMs / 1000} seconds *****");

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = trainedModel.Transform(testingDataView);
            MultiClassClassifierMetrics metrics = context.MulticlassClassification.Evaluate(data: predictions, label: nameof(MNISTData.Number), score: "Score");
            PrintClassifierMetrics(metrics);

            using (FileStream modelStream = new FileStream(model_file, FileMode.OpenOrCreate, FileAccess.Write))
            {
                context.Model.Save(trainedModel, modelStream);
            }

            Console.WriteLine($"The model is saved to {model_file}");
        }

        private static void PrintClassifierMetrics(MultiClassClassifierMetrics metrics)
        {
            Console.WriteLine($"Accurary marco: {metrics.AccuracyMacro}, Accurary micro: {metrics.AccuracyMicro}");
            Console.WriteLine($"Log loss: {metrics.LogLoss}, Top-K: {metrics.TopK}");
        }
    }
}
