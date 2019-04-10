using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
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

    class MNISTDataReader
    {
        static public int ReadMNISTDatabase(int count = 60000)
        {
            FileStream image = File.OpenRead("data/train-images.idx3-ubyte");
            FileStream label = File.OpenRead("data/train-labels.idx1-ubyte");

            BinaryReader image_rd = new BinaryReader(image);
            BinaryReader label_rd = new BinaryReader(label);

            // Check the magic number
            if (image_rd.ReadNonIntelInt32() != 0x00000803) return -1;
            if (label_rd.ReadNonIntelInt32() != 0x00000801) return -2;

            // Read the number of images and labels
            int i_count = image_rd.ReadNonIntelInt32();
            int l_count = label_rd.ReadNonIntelInt32();

            // Check the number of images and labels
            if (i_count != l_count) return -3;
            if (i_count <= 0) return -4;

            // Read the resolution of image
            int image_w = image_rd.ReadNonIntelInt32();
            int image_h = image_rd.ReadNonIntelInt32();

            List<double[]> InImage = new List<double[]>(), OutLabel = new List<double[]>();

            // Starting read
            for (int i = 0; i < i_count; i++)
            {
                //Read the file
                byte cLabel = label_rd.ReadByte();
                //byte[,] cImage = new byte[image_w, image_h];
                double[] cImageFlat = new double[image_w * image_h];

                // Read the image
                int j = 0;
                for (int x = 0; x < image_w; x++)
                {
                    for (int y = 0; y < image_h; y++, j++)
                    {
                        cImageFlat[j] = image_rd.ReadByte();
                        //cImage[x, y] = (byte)cImage[x, y];
                    }
                }

                // Generate label set
                double[] labels = new double[10];
                for (int k = 0; k < 10; k++)
                {
                    if (k == cLabel) labels[k] = 1;
                    else labels[k] = -1;
                }

                // Generate IO set
                InImage.Add(cImageFlat);
                OutLabel.Add(labels);

                if (count == i) break;
            }

            MNISTTrainDataSet = new IOMetaDataSet<double[]>(InImage, OutLabel);

            image_rd.Close();
            label_rd.Close();

            return 0;
        }

        static public int ReadMNISTTestdata(int count = 10000)
        {
            FileStream image = File.OpenRead("data/t10k-images.idx3-ubyte");
            FileStream label = File.OpenRead("data/t10k-labels.idx1-ubyte");

            BinaryReader image_rd = new BinaryReader(image);
            BinaryReader label_rd = new BinaryReader(label);

            // Check the magic number
            if (image_rd.ReadNonIntelInt32() != 0x00000803) return -1;
            if (label_rd.ReadNonIntelInt32() != 0x00000801) return -2;

            // Read the number of images and labels
            int i_count = image_rd.ReadNonIntelInt32();
            int l_count = label_rd.ReadNonIntelInt32();

            // Check the number of images and labels
            if (i_count != l_count) return -3;
            if (i_count <= 0) return -4;

            // Read the resolution of image
            int image_w = image_rd.ReadNonIntelInt32();
            int image_h = image_rd.ReadNonIntelInt32();

            List<double[]> InImage = new List<double[]>(), OutLabel = new List<double[]>();

            // Starting read
            for (int i = 0; i < i_count; i++)
            {
                //Read the file
                byte cLabel = label_rd.ReadByte();
                //byte[,] cImage = new byte[image_w, image_h];
                double[] cImageFlat = new double[image_w * image_h];

                // Read the image
                int j = 0;
                for (int x = 0; x < image_w; x++)
                {
                    for (int y = 0; y < image_h; y++, j++)
                    {
                        cImageFlat[j] = image_rd.ReadByte();
                        //cImage[x, y] = (byte)cImage[x, y];
                    }
                }

                // Generate label set
                double[] labels = new double[10];
                for (int k = 0; k < 10; k++)
                {
                    if (k == cLabel) labels[k] = 1;
                    else labels[k] = -1;
                }

                // Generate IO set
                InImage.Add(cImageFlat);
                OutLabel.Add(labels);

                if (count == i) break;
            }

            MNISTTestDataSet = new IOMetaDataSet<double[]>(InImage, OutLabel);

            image_rd.Close();
            label_rd.Close();

            return 0;
        }
    }
}
