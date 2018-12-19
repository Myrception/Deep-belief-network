using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.MLPComponents
{ 
    public class MLPWeightChange
    {
        private int _Zeile;

        public int Zeile
        {
            get { return _Zeile; }
            set { _Zeile = value; }
        }

        private int _Spalte;

        public int Spalte
        {
            get { return _Spalte; }
            set { _Spalte = value; }
        }

        private double _Änderungswert;

        public double Änderungswert
        {
            get { return _Änderungswert; }
            set { _Änderungswert = value; }
        }
    }
}
