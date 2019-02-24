namespace DeepBeliefNeuralNetwork.MLPComponents
{
    internal class MLPWeightMatrix
    {
        private int _spalten;

        public int Spalten
        {
            get { return _spalten; }
            set { _spalten = value; }
        }

        private int _zeilen;

        public int Zeilen
        {
            get { return _zeilen; }
            set { _zeilen = value; }
        }

        private readonly double[,] _matrix;

        public double this[int x, int y]
        {
            get
            {
                return _matrix[x, y];
            }

            set
            {
                _matrix[x, y] = value;
            }
        }

        public MLPWeightMatrix Laden(string SpeicherortSowieDateiname, MLPWeightMatrix matrix)
        {
            int counter = 0;
            var reader = new System.IO.StreamReader(SpeicherortSowieDateiname);
            while (!reader.EndOfStream)
            {
                string read = reader.ReadLine();
                string[] readsplitted = read.Split(";".ToCharArray());
                for (int i = 0; i < readsplitted.Length - 1; i++)
                {
                    matrix[counter, i] = double.Parse(readsplitted[i]);
                }
                counter++;
            }
            return matrix;
        }

        public void Speichern(string SpeicherortSowieDateiname, MLPWeightMatrix matrix)
        {
            using (System.IO.TextWriter file =
                new System.IO.StreamWriter(@SpeicherortSowieDateiname))//, true
            {
                for (int i = 0; i < matrix.Spalten; i++)
                {
                    for (int j = 0; j < matrix.Zeilen; j++)
                    {
                        file.Write(matrix[i, j] + ";");
                    }
                    file.WriteLine();
                }
                file.Flush();
            }
        }

        public double[,] Clone(MLPWeightMatrix Selber, double[,] Clone)
        {
            double[,] temp = new double[Selber.Zeilen, Selber.Spalten];
            for (int i = 0; i < Selber.Zeilen; i++)
            {
                for (int j = 0; j < Selber.Spalten; j++)
                {
                    temp[i, j] = Selber[i, j];
                }
            }
            return Clone = (double[,])temp.Clone();
        }

        /// <summary>
        /// Der Konstruktor initialisiert die Gewichtsmatrix in der Erforderlichen Größe mit NaN.
        /// </summary>
        /// <param name="spalten">Die Spaltenanzahl ist Eingabeneuron+Ausgabeneuron</param>
        /// <param name="zeilen">Die Zeilenanzahl ist Eingabeneuron+Ausgabeneuron</param>
        public MLPWeightMatrix(int spalten, int zeilen)
        {
            _spalten = spalten;
            _zeilen = zeilen;
            _matrix = new double[zeilen, spalten];
            for (int i = 0; i < _zeilen; i++)
            {
                for (int j = 0; j < _spalten; j++)
                {
                    _matrix[i, j] = 0d; // double.NaN;
                }
            }
        }
    }
}