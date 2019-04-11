using Microsoft.ML.Data;

namespace MNISTTestClient
{
    internal class MNISTData
    {
        [LoadColumn(0)]
        [VectorType(784)]
        public float[] ImageVector;

        [LoadColumn(784)]
        public float Number;
    }

    internal class MNISTNumber
    {
        [ColumnName("Score")]
        [LoadColumn(10)]
        public float[] Score;
    }
}
