function assert(condition, message) {
    if (!condition) {
        throw message || "Assertion failed";
    }
}

function install_dictionary(json) {
    var table = document.getElementById("dictionary");
    for (var i = 0; i < json['dict'].length; ++i) {
        entry = json['dict'][i];
        //console.log(entry);
        var rowCount = table.rows.length;
        var row = table.insertRow(rowCount);
        var layout = [                
            ["x", "dict_xy"],
            ["f", "dict_f"],
            ["i", "dict_i"],                
            ["y", "dict_xy"],
            ["e", "dict_e"],
        ]
        for (var j = 0; j < layout.length; ++j) {
            cell = row.insertCell(j);
            cell.setAttribute("class", layout[j][1]);
            cell.innerHTML = entry[layout[j][0]];
        }
    }
}

function compile_dictionary() {
    var table_data = { 'dict':[] }
    var table = document.getElementById("dictionary");
    var n_rows = table.rows.length;
    var layout = [                
        ["x", "dict_xy"],
        ["f", "dict_f"],
        ["i", "dict_i"],                
        ["y", "dict_xy"],
        ["e", "dict_e"],
    ]
    for (var i = 1; i < n_rows; i++) {
        row_data = {}
        row = table.rows[i];
        for (var j = 0; j < layout.length; ++j) {
            var key = layout[j][0];
            row_data[key] = row.cells[j].innerHTML;
        }
        table_data['dict'].push(row_data);
        //console.log(row_data);
    }
    return table_data;
}

function dictionary_add_row() {
    var table = document.getElementById("dictionary");
    var form = document.getElementById("dictionary_form");
    
    var layout = [                
        ["x", "dict_xy"],
        ["f", "dict_f"],
        ["i", "dict_i"],                
        ["y", "dict_xy"],
        ["e", "dict_e"],
    ]
    
    var entry = {}
    var rowCount = table.rows.length;
    var row = table.insertRow(rowCount);
    
    for (var i = 0; i < layout.length; ++i) {
        key = layout[i][0];
        var item = document.getElementById(key);
        cell = row.insertCell(i);
        cell.setAttribute("class", layout[i][1]);
        cell.innerHTML = item.value;
    }

    document.getElementById("x").value = "";
    document.getElementById("y").value = "";
    document.getElementById("e").value = "";
    document.getElementById("i").value = "";
    document.getElementById("x").focus();
}

function load_dictionary(url) {
    var json = (function () {
            var json = null;
            $.ajax({
                'async': false,
                'global': false,
                'url': url,
                'dataType': "json",
                'success': function (data) {
                    json = data;
                }
            });
            return json;
        })();
    //console.log(json)
    return json;
}

function push_dictionary() {
    var table_data = compile_dictionary()
    var table_json = JSON.stringify(table_data, null, 2);
    $.ajax({
       url: "dictionary.cgi",
       type: 'POST',
       data: { "data" : table_json },
       success: function(response) {
           console.log(response)
           console.log("PUSH SUCCESS")
       }
    });
}

function save_dictionary() {
    var table_data = compile_dictionary()
    var table_json = JSON.stringify(table_data, null, 2);
    
    var blob = new Blob([table_json], {type:'text/json;charset=utf8;'});
    saveAs(blob, "dictionary.json");
    
    var url = URL.createObjectURL( new Blob([table_json], {type:'text/json;charset=utf8;'}) );    
    document.getElementById('dictionary_download').href = url;
    
    return;
}


