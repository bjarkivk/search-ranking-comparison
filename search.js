const http = require("http");
var fs = require("fs");

// Functions

// Writes JSON object obj with elasticsearch results to a file
function write_to_file(obj) {
  fs.writeFile("searchResults.json", JSON.stringify(obj), function (err) {
    if (err) throw err;
    console.log("Search results saved to searchResults.json");
  });
}

// Takes arg = "[a/b/c]" and separates into array [a,b,c]
function seperate(arg) {
  let firstChar = arg.charAt(0);
  let lastChar = arg.charAt(arg.length - 1);
  if (firstChar === "[" && lastChar === "]") {
    str = arg.slice(1, -1);
    const array = str.split("/");
    return array;
  } else {
    throw "Search term not on correct format";
  }
}

const arg = process.argv[2];
searchArray = seperate(arg);
query = searchArray.join(" ");
console.log(query);

// paragraph query
const data = JSON.stringify({
  size: 1000,
  query: {
    match: {
      paragraph: query,
    },
  },
});

const options = {
  hostname: "localhost",
  port: 9200,
  path: "/paragraphs/_search/",
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    // "Content-Length": "data.length",
  },
  body: data,
};

const req = http.request(options, (res) => {
  console.log(`statusCode: ${res.statusCode}`);

  //   res.on("data", (d) => {
  //     process.stdout.write(d);
  //   });

  var str = "";
  res
    .on("data", (d) => {
      str += d;
    })
    .on("end", () => {
      // write to file
      const obj = JSON.parse(str);
      write_to_file(obj);
    });
});

req.on("error", (error) => {
  console.error(error);
});

req.write(data);
req.end();
