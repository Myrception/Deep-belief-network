using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    class ReLu : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            if (input > 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }

        public double BerechneWert(double input, double alpha)
        {
            return Math.Max(0, input);
        }
    }
}
