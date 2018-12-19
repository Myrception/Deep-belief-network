using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    /// <summary>
    /// Diese Klasse stellt eine Schwellwertfunktion zur verfügung.
    /// </summary>
    public class Schwellwertfunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return 0;
        }

        public double BerechneWert(double input, double alpha)
        {
            if (input < 0.5)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new System.NotImplementedException();
        }
    }
}