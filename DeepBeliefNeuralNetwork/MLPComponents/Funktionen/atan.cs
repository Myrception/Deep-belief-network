using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    internal class atan : IFunktionen
    {
        public double BerechneWert(double input, double alpha)
        {
            return Math.Atan(input);
        }

        public double BerechneAbleitung(double input, double alpha)
        {
            return (1 / (Math.Pow(input, 2) + 1));
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}