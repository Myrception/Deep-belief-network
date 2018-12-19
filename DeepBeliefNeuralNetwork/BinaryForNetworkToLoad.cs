using DeepBeliefNeuralNetwork.MLPComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork
{
     public class BinaryForNetworkToLoad
    {
        public BinaryForNetworkToLoad(string DateipfadBinary)
        {
            string Binary;
            using (System.IO.TextReader file =
                new System.IO.StreamReader(DateipfadBinary))//, true
            {
                Binary = file.ReadLine();
            }

            string[] BinarySplit = Binary.Split(@"#".ToCharArray());
            _anzahlLayer = Convert.ToInt32(BinarySplit[0]);
            _neuronsPerLayer = Array.ConvertAll(BinarySplit[1].Split(@":".ToCharArray()), int.Parse); // alternativ int[] myInts = arr.Select(int.Parse).ToArray();
            string[] Aktivierungsfunktion = BinarySplit[2].Split(@":".ToCharArray());
            string[] Outputfunktion = BinarySplit[3].Split(@":".ToCharArray());
            IFunktionen[] aktFu = new IFunktionen[AnzahlLayer];
            IFunktionen[] outFu = new IFunktionen[AnzahlLayer];

            for (int i = 0; i < AnzahlLayer; i++)
            {
                if (Aktivierungsfunktion[i] == "LineareFunktion")
                {
                    aktFu[i] = new MLPComponents.Funktionen.LineareFunktion();
                }
                if (Aktivierungsfunktion[i] == "Tanh")
                {
                    aktFu[i] = new MLPComponents.Funktionen.TangensHyperbolikusFunktion();
                }
                if (Aktivierungsfunktion[i] == "Sigmoide")
                {
                    aktFu[i] = new MLPComponents.Funktionen.SigmoideFunktion();
                }
                if (Outputfunktion[i] == "LineareFunktion")
                {
                    outFu[i] = new MLPComponents.Funktionen.LineareFunktion();
                }
            }
            _aktivierungsFunktion = aktFu;
            _outputFunktion = outFu;
        }


        private int _anzahlLayer;

        public int AnzahlLayer
        {
            get { return _anzahlLayer; }
            set { _anzahlLayer = value; }
        }


        private int[] _neuronsPerLayer;

        public int[] NeuronsPerLayer
        {
            get { return _neuronsPerLayer; }
            set { _neuronsPerLayer = value; }
        }


        private IFunktionen[] _aktivierungsFunktion;

        public IFunktionen[] AktivierungsFunktion
        {
            get { return _aktivierungsFunktion; }
            set { _aktivierungsFunktion = value; }
        }


        private IFunktionen[] _outputFunktion;

        public IFunktionen[] OutputFunktion
        {
            get { return _outputFunktion; }
            set { _outputFunktion = value; }
        }
    }
}
