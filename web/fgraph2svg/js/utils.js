
var utils = {

    load_json: function(url) {
        var j = [];
        $.ajax({
          type: 'GET',
          url: url,
          dataType: 'json',
          success: function(data) { j = data;},
          async: false
        });
        return j;
    },

    request_load_data: function (filename, callfct, args, outfct) {
        console.log("Issue data request: ", filename);
        var xhr = new XMLHttpRequest();
        xhr.open('GET', filename, true);
        xhr.onreadystatechange = function(e) {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    var data = JSON.parse(xhr.responseText);
                    window.data = data;
                    console.log("Processing data from ", filename);
                    returnObj = callfct(data, args);
                    if (outfct) outfct();
                }
            }
        };
        console.log("Finish data request: ", filename);
        xhr.send(null);
    },

    send_data: function(data, url) {
        var data_json = JSON.stringify(data, null, 1);

        $.ajax({
           url: '/getpoint/',
           type: 'GET',
           success: function(response) {
               console.log("Get received.");
               console.log(response);
           }
        });

        $.ajax({
           url: '/endpoint/',
           type: 'POST',
           data: { "data" : data },
           success: function(response) {
               console.log("Post received.");
               console.log(response);
           }
        });
    },

    getTimeStamp: function() {
        /*var newDate = new Date();
        newDate.setTime(Date.now());
        return newDate.toUTCString();*/
        d = new Date();
        return d.toISOString().replace(/T/, ' ').replace(/\..+/, '').replace(/\s/, '_').replace(/:/g,'-');
    },

    centre_div: function(id, prop="left", off=0) {
        $(id).css(prop, function() {
            var w_window = window.innerWidth;
            var w_parent = parseInt($(this).parent().css("width")) - off;
            var w_this = parseInt($(this).css("width"));
            var l_parent = parseInt($(this).parent().css("left"));
            var w_parent_over = w_window - w_parent - l_parent;
            if (w_parent_over < 0) w_parent = w_parent + w_parent_over;
            var l = off + 0.5*(w_parent - w_this)
            return l;
        })
    },

    span_div: function(id, frac, off, prop) {
        $(id).css(prop, function() {
            var w_parent = parseInt($(this).parent().css("width"));
            console.log(w_parent);
            return frac*w_parent - off;
        })
    },

    interpolate_color: function(a, b, amount) {
        var ah = parseInt(a.replace(/#/g, ''), 16),
            ar = ah >> 16, ag = ah >> 8 & 0xff, ab = ah & 0xff,
            bh = parseInt(b.replace(/#/g, ''), 16),
            br = bh >> 16, bg = bh >> 8 & 0xff, bb = bh & 0xff,
            rr = ar + amount * (br - ar),
            rg = ag + amount * (bg - ag),
            rb = ab + amount * (bb - ab);
        return '#' + ((1 << 24) + (rr << 16) + (rg << 8) + rb | 0).toString(16).slice(1);
    },

    round : function(x, prec) {
        return x.toFixed(prec);
    }

};
