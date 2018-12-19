using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
   internal class RBMBiasNeuron
    {
        private double _BiasHidden;

        public double BiasHidden
        {
            get { return _BiasHidden; }
            set { _BiasHidden = value; }
        }
        private double _BiasVisible;

        public double BiasVisible
        {
            get { return _BiasVisible; }
            set { _BiasVisible = value; }
        }
        public RBMBiasNeuron()
        {
            BiasHidden = 0d;
            BiasVisible = 0d;
        }

    }
}
