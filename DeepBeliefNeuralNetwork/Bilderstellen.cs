using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork
{
    /// <summary>
/// Methode zur Erstellung eines Bildes
/// </summary>
    public class Bilderstellen
    {
        private Bitmap bmp;
        /// <summary>
        /// Konstruktor des Objekts
        /// </summary>
        /// <param name="hoehe">Anzahl der Horizontalen Pixel</param>
        /// <param name="breite">Anzahl der Vertikalen Pixel</param>
        public Bilderstellen(int hoehe, int breite)
        {
            bmp = new Bitmap(breite, hoehe);
        }
        /// <summary>
        /// Methode um in dem erstellten Bild ein Pixel zu platzieren
        /// </summary>
        /// <param name="x">X Koordinate</param>
        /// <param name="y">Y Koordinate</param>
        /// <param name="color">Farbe des zu erstellnden Pixels</param>
        public void BildPixelPlazieren(int x, int y ,Color color)
        {
            bmp.SetPixel(x, y, color);
        }
        /// <summary>
        /// Methode zur Speicherung des Bildes
        /// </summary>
        /// <param name="speicherort">Angabe des Speicherortes</param>
        public void SavePicture(string speicherort)
        {
            bmp.Save(speicherort,ImageFormat.Png);
        }
    }
}
