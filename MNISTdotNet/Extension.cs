using System;
using System.IO;

namespace MNISTdotNet
{
    public static class BigEndianUtils
    {
        public static int ReadNonIntelInt32(this BinaryReader br)
        {
            byte[] bytes = br.ReadBytes(sizeof(int));
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }

            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
