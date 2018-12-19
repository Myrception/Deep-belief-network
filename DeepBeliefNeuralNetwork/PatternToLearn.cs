using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork
{
    /// <summary>
    /// Klasse zur Erstellung der Muster die gelernt werden sollen.
    /// </summary>
    public class PatternToLearn
    {


        private double[] _inputvector;

        public double[] inputvector
        {
            get { return _inputvector; }
            set { _inputvector = value; }
        }

        private double[] _targetvector;

        public double[] targetvector
        {
            get { return _targetvector; }
            set { _targetvector = value; }
        }

        private double[] _probabilitie;

        public double[] probabilitie
        {
            get { return _probabilitie; }
            set { _probabilitie = value; }
        }
        private double[] _RBMOutput;

        public double[] RBMOutput
        {
            get { return _RBMOutput; }
            set { _RBMOutput = value; }
        }
    }
}
