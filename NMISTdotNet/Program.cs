using System;
using Microsoft.Data.DataView;
using Microsoft.ML;

namespace NMISTdotNet
{
    class Program
    {
        static void Main(string[] args)
        {
            // Using a 0 seed to initialize the machine learning context
            MLContext context = new MLContext(seed: 0);

            // Create a dataview for training
            IDataView trainingDataView = context.Data.LoadFromTextFile<MNISTData>("mnist_training.dat");

            

        }
    }
}
