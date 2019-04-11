using Microsoft.ML.Data;

namespace MNISTTestClient
{
    internal class MNISTData
    {
        [LoadColumn(0)]
        [VectorType(784)]
        [ColumnName("ImageVector")]
        public float[] ImageVector;

        [LoadColumn(784)]
        public string Number;
    }

    internal class MNISTNumber
    {
        [ColumnName("PredictedLabel")]
        public string Number;
    }
}
