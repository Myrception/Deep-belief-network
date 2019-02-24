using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    /// <summary>
    /// Diese Klasse stellt eine spezielle Rechteckfunktion zur verfügung mit welcher ohne Hidden Layer XOR gelernt werden kann.
    /// Mit entsprechenden änderungen kann auch XNOR gelernt werden.
    /// </summary>
    public class Rechteckfunktion : IFunktionen
    {
        public double BerechneAbleitung(double input, double alpha)
        {
            return 0;
        }

        public double BerechneWert(double input, double alpha)
        {
            if (input < 0.25)
            {
                return 0;
            }
            else if (input > 0.75)
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