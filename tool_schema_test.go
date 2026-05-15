package fugue

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// mustReflect runs reflectInputSchema on sample's type and returns the
// resulting JSON Schema RawMessage. Fails the test on error.
func mustReflect(t *testing.T, sample any) json.RawMessage {
	t.Helper()
	got, err := reflectInputSchema(reflect.TypeOf(sample))
	if err != nil {
		t.Fatalf("reflectInputSchema: %v", err)
	}
	return got
}

func TestReflect_StringField(t *testing.T) {
	type In struct {
		Q string `json:"q" fugue:"the search query"`
	}
	got := mustReflect(t, In{})
	want := `{"type":"object","properties":{"q":{"description":"the search query","type":"string"}},"required":["q"]}`
	if string(got) != want {
		t.Errorf("schema =\n%s\nwant\n%s", got, want)
	}
}

func TestReflect_BoolField(t *testing.T) {
	type In struct {
		Active bool `json:"active"`
	}
	got := mustReflect(t, In{})
	want := `{"type":"object","properties":{"active":{"type":"boolean"}},"required":["active"]}`
	if string(got) != want {
		t.Errorf("schema =\n%s\nwant\n%s", got, want)
	}
}

func TestReflect_IntegerFields(t *testing.T) {
	type In struct {
		A int   `json:"a"`
		B int8  `json:"b"`
		C int64 `json:"c"`
		D uint8 `json:"d"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"a":{"type":"integer"}`) ||
		!strings.Contains(string(got), `"b":{"type":"integer"}`) ||
		!strings.Contains(string(got), `"c":{"type":"integer"}`) ||
		!strings.Contains(string(got), `"d":{"type":"integer"}`) {
		t.Errorf("expected all integer types, got: %s", got)
	}
}

func TestReflect_FloatFields(t *testing.T) {
	type In struct {
		X float32 `json:"x"`
		Y float64 `json:"y"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"x":{"type":"number"}`) ||
		!strings.Contains(string(got), `"y":{"type":"number"}`) {
		t.Errorf("expected float types as number, got: %s", got)
	}
}

func TestReflect_BareUintField(t *testing.T) {
	type In struct {
		Count uint `json:"count"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"count":{"type":"integer"}`) {
		t.Errorf("expected count:{type:integer}, got: %s", got)
	}
}

func TestReflect_NonStructTopLevelRejected(t *testing.T) {
	_, err := reflectInputSchema(reflect.TypeOf("hello"))
	if err == nil {
		t.Fatal("expected error for non-struct top-level type")
	}
	if !strings.Contains(err.Error(), "struct") {
		t.Errorf("error should mention struct, got: %v", err)
	}
}
