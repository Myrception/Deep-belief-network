using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{/// <summary>
/// Diese Klasse stellt eine Lineare Funktion zur verfügung welche bei 1 und -1 in die Sättigung läuft.
/// </summary>
    public class LinearSättigungsFunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            if (input > 1)
            {
                return (0);
            }
            else if (input < -1)
            {
                return (0);
            }
            else
            {
                return 1;
            }
        }

        public double BerechneWert(double input, double alpha)
        {
            if (input > 1)
            {
                return (1);
            }
            else if (input < -1)
            {
                return (-1);
            }
            else
            {
                return input;
            }
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new System.NotImplementedException();
        }
    }
}