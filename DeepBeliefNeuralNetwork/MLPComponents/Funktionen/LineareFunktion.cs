using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    /// <summary>
    /// Diese Klasse stellt eine normale Lineare Funktion zur verfügung mit Steigung von 1.
    /// </summary>
    public class LineareFunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return alpha * 1;
        }

        public double BerechneWert(double input, double alpha)
        {
            return alpha * input;
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new System.NotImplementedException();
        }
    }
}