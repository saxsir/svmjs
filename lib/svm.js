/*!
 * Modified svm.js v1.1.0
 * https://github.com/saxsir/svmjs
 *
 * [Original] https://github.com/karpathy/svmjs
 *
 * Copyright 2013 saxsir
 * Released under the MIT license
 *
 * Date: 2013-10-14
 */

/*
  Copyright (c) 2012 Andrej Karpathy

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// MIT License
// Andrej Karpathy

//@notice: jQuery is needed to function of cross validation
var $ = require('jquery');

var svmjs = (function(exports){

  /*
    This is a binary SVM and is trained using the SMO algorithm.
    Reference: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)
    
    Simple usage example:
    svm = svmjs.SVM();
    svm.train(data, labels);
    testlabels = svm.predict(testdata);
  */
  var SVM = function(options) {
  }

  SVM.prototype = {
    
    // data is NxD array of floats. labels are 1 or -1.
    train: function(data, labels, options) {
      
      // we need these in helper functions
      this.data = data;
      this.labels = labels;

      // parameters
      options = options || {};
      var C = options.C || 1.0; // C value. Decrease for more regularization
      var tol = options.tol || 1e-4; // numerical tolerance. Don't touch unless you're pro
      var alphatol = options.alphatol || 1e-7; // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
      var maxiter = options.maxiter || 10000; // max number of iterations
      var numpasses = options.numpasses || 10; // how many passes over data with no change before we halt? Increase for more precision.
      
      // instantiate kernel according to options. kernel can be given as string or as a custom function
      var kernel = linearKernel;
      this.kernelType = "linear";
      if("kernel" in options) {
        if(typeof options.kernel === "string") {
          // kernel was specified as a string. Handle these special cases appropriately
          if(options.kernel === "linear") { 
            this.kernelType = "linear"; 
            kernel = linearKernel; 
          }
          if(options.kernel === "rbf") { 
            var rbfSigma = options.rbfsigma || 0.5;
            this.rbfSigma = rbfSigma; // back this up
            this.kernelType = "rbf";
            kernel = makeRbfKernel(rbfSigma);
          }
        } else {
          // assume kernel was specified as a function. Let's just use it
          this.kernelType = "custom";
          kernel = options.kernel;
        }
      }

      // initializations
      this.kernel = kernel;
      this.N = data.length; var N = this.N;
      this.D = data[0].length; var D = this.D;
      this.alpha = zeros(N);
      this.b = 0.0;
      this.usew_ = false; // internal efficiency flag

      // Cache kernel computations to avoid expensive recomputation.
      // This could use too much memory if N is large.
      if (options.memoize) {
        this.kernelResults = new Array(N);
        for (var i=0;i<N;i++) {
          this.kernelResults[i] = new Array(N);
          for (var j=0;j<N;j++) {
            this.kernelResults[i][j] = kernel(data[i],data[j]);
          }
        }
      }

      // run SMO algorithm
      var iter = 0;
      var passes = 0;
      while(passes < numpasses && iter < maxiter) {
        
        var alphaChanged = 0;
        for(var i=0;i<N;i++) {
        
          var Ei= this.marginOne(data[i]) - labels[i];
          if( (labels[i]*Ei < -tol && this.alpha[i] < C)
           || (labels[i]*Ei > tol && this.alpha[i] > 0) ){
            
            // alpha_i needs updating! Pick a j to update it with
            var j = i;
            while(j === i) j= randi(0, this.N);
            var Ej= this.marginOne(data[j]) - labels[j];
            
            // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
            ai= this.alpha[i];
            aj= this.alpha[j];
            var L = 0; var H = C;
            if(labels[i] === labels[j]) {
              L = Math.max(0, ai+aj-C);
              H = Math.min(C, ai+aj);
            } else {
              L = Math.max(0, aj-ai);
              H = Math.min(C, C+aj-ai);
            }
            
            if(Math.abs(L - H) < 1e-4) continue;

            var eta = 2*this.kernelResult(i,j) - this.kernelResult(i,i) - this.kernelResult(j,j);
            if(eta >= 0) continue;
            
            // compute new alpha_j and clip it inside [0 C]x[0 C] box
            // then compute alpha_i based on it.
            var newaj = aj - labels[j]*(Ei-Ej) / eta;
            if(newaj>H) newaj = H;
            if(newaj<L) newaj = L;
            if(Math.abs(aj - newaj) < 1e-4) continue; 
            this.alpha[j] = newaj;
            var newai = ai + labels[i]*labels[j]*(aj - newaj);
            this.alpha[i] = newai;
            
            // update the bias term
            var b1 = this.b - Ei - labels[i]*(newai-ai)*this.kernelResult(i,i)
                     - labels[j]*(newaj-aj)*this.kernelResult(i,j);
            var b2 = this.b - Ej - labels[i]*(newai-ai)*this.kernelResult(i,j)
                     - labels[j]*(newaj-aj)*this.kernelResult(j,j);
            this.b = 0.5*(b1+b2);
            if(newai > 0 && newai < C) this.b= b1;
            if(newaj > 0 && newaj < C) this.b= b2;
            
            alphaChanged++;
            
          } // end alpha_i needed updating
        } // end for i=1..N
        
        iter++;
        //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
        if(alphaChanged == 0) passes++;
        else passes= 0;
        
      } // end outer loop
      
      // if the user was using a linear kernel, lets also compute and store the
      // weights. This will speed up evaluations during testing time
      if(this.kernelType === "linear") {

        // compute weights and store them
        this.w = new Array(this.D);
        for(var j=0;j<this.D;j++) {
          var s= 0.0;
          for(var i=0;i<this.N;i++) {
            s+= this.alpha[i] * labels[i] * data[i][j];
          }
          this.w[j] = s;
          this.usew_ = true;
        }
      } else {

        // okay, we need to retain all the support vectors in the training data,
        // we can't just get away with computing the weights and throwing it out

        // But! We only need to store the support vectors for evaluation of testing
        // instances. So filter here based on this.alpha[i]. The training data
        // for which this.alpha[i] = 0 is irrelevant for future. 
        var newdata = [];
        var newlabels = [];
        var newalpha = [];
        for(var i=0;i<this.N;i++) {
          //console.log("alpha=%f", this.alpha[i]);
          if(this.alpha[i] > alphatol) {
            newdata.push(this.data[i]);
            newlabels.push(this.labels[i]);
            newalpha.push(this.alpha[i]);
          }
        }

        // store data and labels
        this.data = newdata;
        this.labels = newlabels;
        this.alpha = newalpha;
        this.N = this.data.length;
        //console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
      }

      var trainstats = {};
      trainstats.iters= iter;
      return trainstats;
    }, 
    
    // inst is an array of length D. Returns margin of given example
    // this is the core prediction function. All others are for convenience mostly
    // and end up calling this one somehow.
    marginOne: function(inst) {

      var f = this.b;
      // if the linear kernel was used and w was computed and stored,
      // (i.e. the svm has fully finished training)
      // the internal class variable usew_ will be set to true.
      if(this.usew_) {

        // we can speed this up a lot by using the computed weights
        // we computed these during train(). This is significantly faster
        // than the version below
        for(var j=0;j<this.D;j++) {
          f += inst[j] * this.w[j];
        }

      } else {

        for(var i=0;i<this.N;i++) {
          f += this.alpha[i] * this.labels[i] * this.kernel(inst, this.data[i]);
        }
      }

      return f;
    },
    
    predictOne: function(inst) { 
      return this.marginOne(inst) > 0 ? 1 : -1; 
    },
    
    // data is an NxD array. Returns array of margins.
    margins: function(data) {
      
      // go over support vectors and accumulate the prediction. 
      var N = data.length;
      var margins = new Array(N);
      for(var i=0;i<N;i++) {
        margins[i] = this.marginOne(data[i]);
      }
      return margins;
      
    },

    kernelResult: function(i, j) {
      if (this.kernelResults) {
        return this.kernelResults[i][j];
      }
      return this.kernel(this.data[i], this.data[j]);
    },

    // data is NxD array. Returns array of 1 or -1, predictions
    predict: function(data) {
      var margs = this.margins(data);
      for(var i=0;i<margs.length;i++) {
        margs[i] = margs[i] > 0 ? 1 : -1;
      }
      return margs;
    },
    
    // THIS FUNCTION IS NOW DEPRECATED. WORKS FINE BUT NO NEED TO USE ANYMORE. 
    // LEAVING IT HERE JUST FOR BACKWARDS COMPATIBILITY FOR A WHILE.
    // if we trained a linear svm, it is possible to calculate just the weights and the offset
    // prediction is then yhat = sign(X * w + b)
    getWeights: function() {
      
      // DEPRECATED
      var w= new Array(this.D);
      for(var j=0;j<this.D;j++) {
        var s= 0.0;
        for(var i=0;i<this.N;i++) {
          s+= this.alpha[i] * this.labels[i] * this.data[i][j];
        }
        w[j]= s;
      }
      return {w: w, b: this.b};
    },

    toJSON: function() {
      
      if(this.kernelType === "custom") {
        console.log("Can't save this SVM because it's using custom, unsupported kernel...");
        return {};
      }

      json = {}
      json.N = this.N;
      json.D = this.D;
      json.b = this.b;

      json.kernelType = this.kernelType;
      if(this.kernelType === "linear") { 
        // just back up the weights
        json.w = this.w; 
      }
      if(this.kernelType === "rbf") { 
        // we need to store the support vectors and the sigma
        json.rbfSigma = this.rbfSigma; 
        json.data = this.data;
        json.labels = this.labels;
        json.alpha = this.alpha;
      }

      return json;
    },
    
    fromJSON: function(json) {
      
      this.N = json.N;
      this.D = json.D;
      this.b = json.b;

      this.kernelType = json.kernelType;
      if(this.kernelType === "linear") { 

        // load the weights! 
        this.w = json.w; 
        this.usew_ = true; 
        this.kernel = linearKernel; // this shouldn't be necessary
      }
      else if(this.kernelType == "rbf") {

        // initialize the kernel
        this.rbfSigma = json.rbfSigma; 
        this.kernel = makeRbfKernel(this.rbfSigma);

        // load the support vectors
        this.data = json.data;
        this.labels = json.labels;
        this.alpha = json.alpha;
      } else {
        console.log("ERROR! unrecognized kernel type." + this.kernelType);
      }
    },

    // saxsir added 2013.10.14
    crossValidation: function(_data, _labels, opts, k) {
      //@notice: Show num of subsets
      console.log('K is ' + k);
      //@notice: _dataと_labelsは関数内で削除されるので値渡しする
      var data = $.extend(true, [], _data),
        labels = $.extend(true, [], _labels);

      //@debug
      /*
      console.log('### data ###');
      console.log(data);
      console.log('### /data ### ');
      console.log('### labels ###');
      console.log(labels);
      console.log('### /labels ###');
      */

      //@notice: サブセットクラス
      var Subset = function() {
        this.data = [];
        this.labels = [];
      };

      //@notice: データセットをk個のサブセットに分割する
      var subsets = [];
      for (var i = 0; i < k; i++) {
        subsets.push(new Subset());
      }

      //@notice: ランダム、かつ均等に分割する
      for (var i = 0, il = data.length; i < il; i++) {
        var group = i % k; // 0 ~ k-1
        //@notice: data.lengthは毎回変わるので変数に代入不可
        var rnd = Math.random() * data.length | 0; //0 ~ data.length-1 
        var d = data.splice(rnd, 1); // [[x,y]]
        var l = labels.splice(rnd, 1); // [x]
        subsets[group].data.push(d[0]);
        subsets[group].labels.push(l[0]);
      }


      //@notice: 1つのサブセットをテスト用にして、残りでsvmをトレーニングする
      //  これをk回（各サブセット分）繰り返す
      //  k回分の検証結果の配列を返す
      var hit = 0;
      for (var i = 0, il = subsets.length; i < il; i++) {
        //@notice: Check runnning process
        console.log(i + ': Running Cross Validation...');

        //@notice: 値渡しで配列をコピー、subsets[i]はテストデータとして利用
        var subsetsCopy = $.extend(true, [], subsets),
          subset = subsetsCopy.splice(i, 1)[0],
          testData = subset.data,
          testLabels = subset.labels;

        //@notice: 残りはトレーニング用データ
        var trainingData = [], trainingLabels = [];
        for (var j = 0, jl = subsetsCopy.length; j < jl; j++) {
          Array.prototype.push.apply(trainingData, subsetsCopy[j].data);
          Array.prototype.push.apply(trainingLabels, subsetsCopy[j].labels);
        }

        //@debug
        /*
        console.log('### testData ###');
        console.log(testData);
        console.log('### /testData ### ');
        console.log('### testLabels ###');
        console.log(testLabels);
        console.log('### /testLabels ###');

        //@debug
        console.log('### trainingData ###');
        console.log(trainingData);
        console.log('### /trainingData ### ');
        console.log('### trainingLabels ###');
        console.log(trainingLabels);
        console.log('### /trainingLabels ###');
        */

        //@notice: 分類器のトレーニング
        var svm = new SVM();
        svm.train(trainingData, trainingLabels, opts);

        //@notice: テスト用データを使って予想が合っているか正答数を確認
        var forecasts = svm.predict(testData);
        for (var j = 0, jl = forecasts.length; j < jl; j++) {
          var forecast = forecasts[j], answer = testLabels[j];
          //@debug
          /*
          console.log(forecast + ' === ' + answer);
          console.log(forecast === answer);
          */
          if (forecast === answer) {
            hit++;
          }
        }
      }
      // return (num of all data) - (num of correct answers)
      // return hit;
      //@notice: 正答率を返す
      return (hit / _data.length) * 100;
    }
  }
  
  // Kernels
  function makeRbfKernel(sigma) {
    return function(v1, v2) {
      var s=0;
      for(var q=0;q<v1.length;q++) { s += (v1[q] - v2[q])*(v1[q] - v2[q]); } 
      return Math.exp(-s/(2.0*sigma*sigma));
    }
  }
  
  function linearKernel(v1, v2) {
    var s=0; 
    for(var q=0;q<v1.length;q++) { s += v1[q] * v2[q]; } 
    return s;
  }

  // Misc utility functions
  // generate random floating point number between a and b
  function randf(a, b) {
    return Math.random()*(b-a)+a;
  }

  // generate random integer between a and b (b excluded)
  function randi(a, b) {
     return Math.floor(Math.random()*(b-a)+a);
  }

  // create vector of zeros of length n
  function zeros(n) {
    var arr= new Array(n);
    for(var i=0;i<n;i++) { arr[i]= 0; }
    return arr;
  }

  // export public members
  exports = exports || {};
  exports.SVM = SVM;
  exports.makeRbfKernel = makeRbfKernel;
  exports.linearKernel = linearKernel;
  return exports;

})(typeof module != 'undefined' && module.exports);  // add exports to module.exports if in node.js
