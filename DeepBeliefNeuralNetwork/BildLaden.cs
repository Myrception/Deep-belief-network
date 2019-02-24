using System.Drawing;

namespace DeepBeliefNeuralNetwork
{
    public class BildLaden
    {
        private Bitmap bmp;

        public Bitmap LoadPicture(string picturePath)
        {
            return bmp = (Bitmap)Image.FromFile(picturePath);
        }
    }
}