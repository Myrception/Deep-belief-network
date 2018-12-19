using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.MLPComponents.Funktionen
{
    class TestFunktion:IFunktionen
    {
        /// <summary>
        /// Experimentele Funktion. Sehr steil um den Nullpunkt
        /// </summary>
        public double BerechneWert(double input, double alpha)
        {
            if (input > 0d)
            {
                return (Math.Sqrt(Math.Tanh(input)));
            }
            else
            {
                return (-(Math.Sqrt(Math.Tanh(Math.Abs(input)))));
            }
        }

        public double BerechneAbleitung(double input, double alpha)
        {
            if (input > 0d)
            {
                return ((1/(2*Math.Sqrt(Math.Tanh(input))))*(1/Math.Pow(Math.Cosh(input),2)));
            }
            else
            {
                return (-(input/(2*Math.Abs(input)*Math.Sqrt(Math.Tanh(Math.Abs(input)))))*(1 / Math.Pow(Math.Cosh(input), 2)));
            }
        }

        public double BerechneAbleitung(double input, double alpha, List<List<MLPNeuron>> layers)
        {
            throw new NotImplementedException();
        }
    }
}
