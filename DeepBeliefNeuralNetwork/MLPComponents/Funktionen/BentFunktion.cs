using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    internal class BentFunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return (((Math.Sqrt(Math.Pow(input, 2d) + 1d) - 1d) / 2d) + input);
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }

        public double BerechneWert(double input, double alpha)
        {
            return ((input / (2 * Math.Sqrt(Math.Pow(input, 2d) + 1d))) + 1d);
        }
    }
}