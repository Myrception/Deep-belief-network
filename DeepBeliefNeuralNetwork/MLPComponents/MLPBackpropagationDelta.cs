using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.MLPComponents
{
    internal class MLPBackpropagationDelta
    {
        private int _Index;

        public int Index
        {
            get { return _Index; }
            set { _Index = value; }
        }

        private double _Delta;

        public double Delta
        {
            get { return _Delta; }
            set { _Delta = value; }
        }
    }
}
