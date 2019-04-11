using Microsoft.ML.Data;

namespace MNISTTestClient
{
    internal class MNISTData
    {
        [LoadColumn(0)]
        [VectorType()]
        public float[] ImageVector;

        [LoadColumn(784)]
        public float Number;
    }

    internal class MNISTNumber
    {
        [ColumnName("Number")]
        public int Number;
    }
}
