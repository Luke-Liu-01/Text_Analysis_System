function choose() {
    var brand = $('#brand').val()
    var model = $('#model').val()

    window.location.href = '?brand=' + brand + '&model=' + model;
}