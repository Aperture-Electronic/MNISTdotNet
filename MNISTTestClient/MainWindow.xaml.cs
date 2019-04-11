using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.ML;

namespace MNISTTestClient
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        private const string ml_model_file = "model/ml-model.mld";

        public MainWindow()
        {
            InitializeComponent();
        }

        private void CanDraw_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                System.Windows.Point point = e.GetPosition(canDraw);
                RectangleGeometry pt = new RectangleGeometry(new Rect(point, new System.Windows.Size(5, 5)));
                System.Windows.Shapes.Path path = new System.Windows.Shapes.Path
                {
                    Stroke = System.Windows.Media.Brushes.Black,
                    StrokeThickness = 5,
                    Data = pt
                };
                canDraw.Children.Add(path);
            }
        }

        private void BtnClear_Click(object sender, RoutedEventArgs e)
        {
            canDraw.Children.Clear();
        }

        private void BtnRegress_Click(object sender, RoutedEventArgs e)
        {
            // Get and convert the bitmap drawed
            // Render the bitmap
            RenderTargetBitmap renderBitmap = new RenderTargetBitmap((int)canDraw.ActualWidth, (int)canDraw.ActualHeight, 96, 96, PixelFormats.Default);
            renderBitmap.Render(canDraw);

            // Convert it to bitmap object
            Bitmap bitmap = null;
            using (MemoryStream stream = new MemoryStream())
            {
                BitmapEncoder encoder = new BmpBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(renderBitmap));
                encoder.Save(stream);
                bitmap = new Bitmap(stream);
            }

            // Resize the bitmap to MNIST image size (28 x 28 = 784 pixels)
            Bitmap MNIST_bitmap = new Bitmap(28, 28);
            using (Graphics graphics = Graphics.FromImage(bitmap))
            {
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(MNIST_bitmap, new Rectangle(0, 0, 28, 28), new Rectangle(0, 0, bitmap.Width, bitmap.Height), GraphicsUnit.Pixel);
            }

            // Read and convert the pixels into ML.net data array
            float[] pixels = new float[28 * 28];
            int p_flat = 0;
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    pixels[p_flat] = (MNIST_bitmap.GetPixel(x, y).R > 0) ? 255 : 0;
                }
            }

            MNISTData data = new MNISTData()
            {
                ImageVector = pixels,
            };

            // Create the machine learning framework (context) and load the trained model
            MLContext context = new MLContext();
            FileStream modelFileStream = new FileStream(ml_model_file, FileMode.Open);
            ITransformer trainedModel = context.Model.Load(modelFileStream);

            // Create prediction engine related to the loaded trained model
            PredictionEngine<MNISTData, MNISTNumber> predEngine = context.Model.CreatePredictionEngine<MNISTData, MNISTNumber>(trainedModel);

            //Input the user data
            MNISTNumber result = predEngine.Predict(data);

            // Get the maximun guess
            int n = 0;
            float max = result.Score[0];
            for (int i = 0; i < 10; i++)
            {
                if(result.Score[i] > max)
                {
                    n = i;
                    max = result.Score[i];
                }

                Console.WriteLine($"Result of [{i}] = {result.Score[i]}");
            }

            // Output the result
            lblResult.Content = n.ToString();
        }
    }
}
