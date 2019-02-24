using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    internal class TanhReLu : IFunktionen
    {
        public double BerechneWert(double input, double alpha)
        {
            return Math.Min(Math.Max(input / 7.1d, -1d), 1d);
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