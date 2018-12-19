using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork.RBMComponents
{
    public class RBMCreateClass
    {

        private int _numberOfVisualNeurons;

        public int numberOfVisualNeurons
        {
            get { return _numberOfVisualNeurons; }
            set { _numberOfVisualNeurons = value; }
        }

        private int _numberOfHiddenNeurons;

        public int numberofHiddenNeurons
        {
            get { return _numberOfHiddenNeurons; }
            set { _numberOfHiddenNeurons = value; }
        }
    }
}
