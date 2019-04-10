using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace NMISTdotNet
{
    class MNISTData
    {
        [LoadColumn(0)]
        public byte[] ImageVector;
    }

    class MNISNumber
    {
        [ColumnName("Number")]
        public int Number;
    }
}
