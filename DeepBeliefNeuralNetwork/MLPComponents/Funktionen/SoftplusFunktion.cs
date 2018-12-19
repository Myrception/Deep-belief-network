using System;
using System.Collections.Generic;
using DeepBeliefNeuralNetwork.MLPComponents;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    /// <summary>
    /// Diese Klasse stellt die Funktion Softplus zur verfügung auch Rectifier genannt. Die Ableitung der Softplus Funktion ist die Sigmoide.
    /// </summary>
    public class SoftplusFunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return (1 / (1 + Math.Exp(-input)));
        }

        public double BerechneWert(double input, double alpha)
        {
            return Math.Log(1 + Math.Exp(input));
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}