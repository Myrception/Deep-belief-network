using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork
{
    public class BildLaden
    {
        private Bitmap bmp;

        public Bitmap LoadPicture(string picturePath)
        {
            return bmp = (Bitmap) Image.FromFile(picturePath);
        }
    }
}
