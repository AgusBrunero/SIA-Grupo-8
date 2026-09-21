Perceptron.py lo hice originalmente en Java y después lo pasé a Python. Código original:

    import java.util.ArrayList;
    import java.util.List;
    import java.util.Random;
    import java.util.function.Function;
    
    public class Neuron {
    private final List<Double> weights;
    private final Function<Double,Double> activationFunction;
    private final Function<Double,Double> derivativeFunction;
    private final double learningRate;

        public Neuron(double sizeOfWeights, Function<Double,Double> activationFunction, Function<Double,Double> derivativeFunction, double learningRate) {
            this.activationFunction = activationFunction;
            this.derivativeFunction = derivativeFunction;
            this.learningRate = learningRate;
            Random rand = new Random(System.currentTimeMillis());
            this.weights = new ArrayList<>();
            // weights[sizeOfWeights] = bias
            for (int i = 0; i <= sizeOfWeights; i++) {
                this.weights.add(rand.nextDouble());
            }
        }
    
        public double compute(List<Double> input){
            if(input.size() != weights.size() - 1) throw new IllegalArgumentException("Inputs must have same size");
            double sum = 0.0;
            for (int i = 0; i < input.size(); i++) {
                sum+= input.get(i)*weights.get(i);
            }
            sum+=weights.getLast();
            return sum;
        }
    
        public double activate(List<Double> input){
            return activationFunction.apply(compute(input));
        }
    
        public void learnOnline(List<List<Double>> inputs, List<Double> expected){
            if(inputs.size() != expected.size()) throw new IllegalArgumentException("Inputs must have same size");
            for(int i = 0; i < inputs.size(); i++) {
                List<Double> input = inputs.get(i);
                double h = compute(input);
                double factor = learningRate*(expected.get(i)-activationFunction.apply(h))*derivativeFunction.apply(h);
                for(int j = 0; j < input.size(); j++)
                    weights.set(j,weights.get(j)+factor*input.get(j));
                double bias = weights.getLast();
                bias+=factor;
                weights.set(weights.size()-1,bias);
            }
        }
    
        public void learnBatch(List<List<Double>> inputs, List<Double> expected){
            if(inputs.size() != expected.size()) throw new IllegalArgumentException("Inputs must have same size");
            List<Double> deltaWeights = new ArrayList<>();
            for(int i = 0; i < inputs.size(); i++) {
                List<Double> input = inputs.get(i);
                double h = compute(input);
                double factor = learningRate*(expected.get(i)-activationFunction.apply(h))*derivativeFunction.apply(h);
                for(int j = 0; j < input.size(); j++)
                    if(i==0)
                        deltaWeights.add(factor*input.get(j));
                    else
                        deltaWeights.set(j,factor*input.get(j));
                if(i==0)
                    deltaWeights.add(factor);
                else
                    deltaWeights.set(deltaWeights.size()-1,deltaWeights.getLast()+factor);
            }
            for(int i = 0; i < deltaWeights.size(); i++)
                weights.set(i,weights.get(i)+deltaWeights.get(i));
        }
    
        public double getError(List<List<Double>> inputs, List<Double> expected){
            if(inputs.size() != expected.size()) throw new IllegalArgumentException("Inputs must have same size");
            double sum = 0.0;
            for(int i = 0; i < inputs.size(); i++)
                sum+=Math.pow(expected.get(i)-activate(inputs.get(i)),2);
            return sum/2;
        }
    }
