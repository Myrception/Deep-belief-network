using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    /// <summary>
    /// Diese Klasse stellt die Funktion Sigmoide zur verfügung.
    /// </summary>
    public class SigmoideFunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return alpha * (1 / (1 + Math.Exp(-input))) * (1 - (1 / (1 + Math.Exp(-input))));
        }

        public double BerechneWert(double input, double alpha)
        {
            return alpha * (1 / (1 + Math.Exp(-input)));
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}