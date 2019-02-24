using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    internal class Sinus : IFunktionen
    {
        public double BerechneWert(double input, double alpha)
        {
            return Math.Sin(input);
        }

        public double BerechneAbleitung(double input, double alpha)
        {
            return Math.Cos(input);
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}