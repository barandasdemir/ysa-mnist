class Aktivasyon {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

const sigmoid = new Aktivasyon(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

const tanh = new Aktivasyon(
  x => Math.tanh(x),
  y => 1 - (y * y)
);


class NeuralNetwork {
  /*
  * if first argument is a NeuralNetwork the constructor clones it
  * USAGE: cloned_nn = new NeuralNetwork(to_clone_nn);
  */
  constructor(giris, gizli, cikis) {
    if (giris instanceof NeuralNetwork) {
      let a = giris;
      this.giris_dugumler = a.giris_dugumler;
      this.gizli_dugumler = a.gizli_dugumler;
      this.cikis_dugumler = a.cikis_dugumler;

      this.agirlik_gg = a.agirlik_gg.copy();
      this.agirlik_gc = a.agirlik_gc.copy();

      this.bias_g = a.bias_g.copy();
      this.bias_c = a.bias_c.copy();
    } else {
      this.giris_dugumler = giris;
      this.gizli_dugumler = gizli;
      this.cikis_dugumler = cikis;

      this.agirlik_gg = new Matrix(this.gizli_dugumler, this.giris_dugumler);
      this.agirlik_gc = new Matrix(this.cikis_dugumler, this.gizli_dugumler);
      this.agirlik_gg.randomize();
      this.agirlik_gc.randomize();

      this.bias_g = new Matrix(this.gizli_dugumler, 1);
      this.bias_c = new Matrix(this.cikis_dugumler, 1);
      this.bias_g.randomize();
      this.bias_c.randomize();
    }

    this.ogrenmeOraniAyarla();
    this.aktivasyonFnAyarla();
  }

  tahmin(input_array) {
    let giris = Matrix.fromArray(input_array);
    // gizli düğümleri oluştur
    let gizli = Matrix.multiply(this.agirlik_gg, giris);
    gizli.add(this.bias_g);
    // aktivasyon fonksiyonunu uygula
    gizli.map(this.aktivasyonFn.func);

    // çıkışları hesapla
    let cikis = Matrix.multiply(this.agirlik_gc, gizli);
    cikis.add(this.bias_c);
    cikis.map(this.aktivasyonFn.func);

    return cikis.toArray();
  }

  ogrenmeOraniAyarla(oran = 0.1) {
    this.ogrenme_orani = oran;
  }

  aktivasyonFnAyarla(fn = sigmoid) {
    this.aktivasyonFn = fn;
  }

  egit(input_array, hedef_arr) {
    let giris = Matrix.fromArray(input_array);
    // gizli düğümleri oluştur
    let gizli = Matrix.multiply(this.agirlik_gg, giris);
    gizli.add(this.bias_g);
    // aktivasyon fonksiyonunu uygula
    gizli.map(this.aktivasyonFn.func);

    // çıkışları hesapla
    let cikis = Matrix.multiply(this.agirlik_gc, gizli);
    cikis.add(this.bias_c);
    cikis.map(this.aktivasyonFn.func);

    let hedef = Matrix.fromArray(hedef_arr);

    // hata oranını hesapla
    // hata = hedef - çıkış
    let cikis_hata = Matrix.subtract(hedef, cikis);

    let egim = Matrix.map(cikis, this.aktivasyonFn.dfunc);
    egim.multiply(cikis_hata);
    egim.multiply(this.ogrenme_orani);


    // deltaları hesapla
    let gizli_T = Matrix.transpose(gizli);
    let agirlik_gc_delta = Matrix.multiply(egim, gizli_T);

    // ağırlıkları deltalara göre ayarla
    this.agirlik_gc.add(agirlik_gc_delta);
    this.bias_c.add(egim);

    // gizli katman hata oranını hesapla
    let who_tp = Matrix.transpose(this.agirlik_gc);
    let gizli_hata = Matrix.multiply(who_tp, cikis_hata);

    // Calculate hidden gradient
    let gizli_egim = Matrix.map(gizli, this.aktivasyonFn.dfunc);
    gizli_egim.multiply(gizli_hata);
    gizli_egim.multiply(this.ogrenme_orani);

    let inputs_T = Matrix.transpose(giris);
    let weight_ih_deltas = Matrix.multiply(gizli_egim, inputs_T);

    this.agirlik_gg.add(weight_ih_deltas);
    this.bias_g.add(gizli_egim);

    // outputs.print();
    // targets.print();
    // error.print();
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
    nn.agirlik_gg = Matrix.deserialize(data.weights_ih);
    nn.agirlik_gc = Matrix.deserialize(data.weights_ho);
    nn.bias_g = Matrix.deserialize(data.bias_h);
    nn.bias_c = Matrix.deserialize(data.bias_o);
    nn.ogrenme_orani = data.learning_rate;
    return nn;
  }

  copy() {
    return new NeuralNetwork(this);
  }

  mutate(func) {
    this.agirlik_gg.map(func);
    this.agirlik_gc.map(func);
    this.bias_g.map(func);
    this.bias_c.map(func);
  }
}
