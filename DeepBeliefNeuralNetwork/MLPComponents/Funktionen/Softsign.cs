using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    class Softsign: IFunktionen
    {
        public double BerechneWert(double input, double alpha)
        {
            return (input / (1 + Math.Abs(input)));
        }

        public double BerechneAbleitung(double input, double alpha)
        {
            return (1 / Math.Pow((1 + Math.Abs(input)), 2));
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}
