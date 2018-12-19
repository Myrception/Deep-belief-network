using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
    internal class RBMWeightMatrix
    {
        public int Row { get; set; }

        public int Coloumn { get; set; }

        private double[,] _matrix;
        /// <summary>
        /// x = row y = Coloumn
        /// </summary>
        /// <param name="x">row</param>
        /// <param name="y">coloum</param>
        /// <returns></returns>
        public double this[int x, int y]
        {
            get { return _matrix[x, y]; }

            set { _matrix[x, y] = value; }
        }
        public void Speichern(string SpeicherortSowieDateiname, RBMWeightMatrix matrix)
        {
            using (System.IO.TextWriter file =
                new System.IO.StreamWriter(@SpeicherortSowieDateiname))//, true
            {
                for (int i = 0; i < matrix.Row; i++)
                {
                    for (int j = 0; j < matrix.Coloumn; j++)
                    {
                        file.Write(matrix[i, j] + ";");
                    }
                    file.WriteLine();
                }
                file.Flush();
            }
        }
        public double[,] Clone(RBMWeightMatrix Selber, double[,] Clone)
        {
            double[,] temp = new double[Selber.Row, Selber.Coloumn];
            for (int i = 0; i < Selber.Row; i++)
            {
                for (int j = 0; j < Selber.Coloumn; j++)
                {
                    temp[i, j] = Selber[i, j];
                }
            }
            return Clone = (double[,])temp.Clone();
        }

        public RBMWeightMatrix(int row, int coloumn)
        {
            Row = row;
            Coloumn = coloumn;
            _matrix = new double[row, coloumn];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < coloumn; j++)
                {
                    _matrix[i, j] = 0;
                }
            }
        }
    }
}
