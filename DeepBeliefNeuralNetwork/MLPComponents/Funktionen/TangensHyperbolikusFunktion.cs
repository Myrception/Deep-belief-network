using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    /// <summary>
    /// Diese Klasse stellt die Funktion Tangens Hyperbolikus zur verfügung.
    /// </summary>
    public class TangensHyperbolikusFunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return (1 - Math.Pow(Math.Tanh(input), 2));
        }

        public double BerechneWert(double input, double alpha)
        {
            return Math.Tanh(input);
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}