using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    internal class ReLu6 : IFunktionen
    {
        public double BerechneWert(double input, double alpha)
        {
            return Math.Min(Math.Max(input, 0), 6);
        }

        public double BerechneAbleitung(double input, double alpha)
        {
            throw new NotImplementedException();
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}